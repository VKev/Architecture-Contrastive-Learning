import argparse
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualise correlations between the input kernels that feed a single "
            "output channel of a convolutional layer."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Path to a torch checkpoint or state_dict file.")
    parser.add_argument(
        "--compare-ckpt",
        type=str,
        help="Optional second checkpoint for comparison against the primary checkpoint.",
    )
    parser.add_argument(
        "--mode",
        choices=["channel", "all-channel", "all-layer"],
        default="channel",
        help="Type of visualisation: per output channel, per layer across all channels, or all-layer (coming soon).",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        help="Index of the convolutional layer to inspect (0-based, negative values index from the end).",
    )
    parser.add_argument(
        "--output-channel",
        type=int,
        help="Output channel index (required for --mode channel; ignored otherwise).",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List convolutional layers discovered in the checkpoint and exit.",
    )
    parser.add_argument("--save", type=str, help="Optional output path for the heatmap (defaults to a derived name).")
    parser.add_argument("--show", action="store_true", help="Display the heatmap window.")
    return parser.parse_args()


def load_state_dict(ckpt_path: str) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, Any]]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    metadata: Dict[str, Any] = {}
    visited: set = set()

    def record(source: Dict[str, Any]) -> None:
        model_keys = ("model", "model_name", "architecture", "arch", "network", "net", "backbone")
        for key in model_keys:
            value = source.get(key)
            if isinstance(value, str) and "model" not in metadata:
                metadata["model"] = value
                break

        contrastive_keys = (
            "contrastive_kernel_loss",
            "contrastive_loss",
            "calculate_contrastive_loss",
            "contrastive_linear_loss",
            "use_contrastive",
            "use_ckl",
        )
        for key in contrastive_keys:
            if metadata.get("contrastive") is True:
                break
            bool_val = _coerce_bool(source.get(key))
            if bool_val is not None:
                if bool_val:
                    metadata["contrastive"] = True
                elif "contrastive" not in metadata:
                    metadata["contrastive"] = False

    def inspect(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            record(obj)
            for value in obj.values():
                inspect(value)
        elif hasattr(obj, "__dict__"):
            inspect(vars(obj))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                inspect(item)

    inspect(checkpoint)

    state_candidate: Optional[Any] = None

    if isinstance(checkpoint, OrderedDict):
        state_candidate = checkpoint
    elif isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "network", "net", "model"):
            value = checkpoint.get(key)
            if isinstance(value, (OrderedDict, dict)):
                state_candidate = value
                break
        if state_candidate is None:
            state_candidate = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    if not isinstance(state_candidate, (OrderedDict, dict)):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping.")

    state_dict = OrderedDict((k, v) for k, v in state_candidate.items() if isinstance(v, torch.Tensor))

    if not state_dict:
        raise KeyError("No tensor weights found in the checkpoint.")

    return state_dict, metadata


def collect_conv_layers(state_dict: "OrderedDict[str, torch.Tensor]") -> List[Tuple[str, torch.Tensor]]:
    conv_layers: List[Tuple[str, torch.Tensor]] = []
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            if tensor.shape[2] == 1 and tensor.shape[3] == 1:
                continue
            conv_layers.append((name, tensor))
    return conv_layers


def print_conv_layers(conv_layers: List[Tuple[str, torch.Tensor]]) -> None:
    if not conv_layers:
        print("No convolutional weight tensors (4D) were found in the provided checkpoint.")
        return

    print("Discovered convolutional weight tensors:\n")
    for idx, (name, tensor) in enumerate(conv_layers):
        out_channels, in_channels, kernel_height, kernel_width = tensor.shape
        print(
            f"[{idx}] {name}: "
            f"out_channels={out_channels}, in_channels={in_channels}, kernel_size=({kernel_height}, {kernel_width})"
        )


def normalise_index(index: int, length: int, label: str) -> int:
    if index < 0:
        index = length + index
    if index < 0 or index >= length:
        raise IndexError(f"{label} {index} is out of range for length {length}.")
    return index


def compute_kernel_correlation(weight: torch.Tensor, output_channel: int) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError("Expected a 4D convolutional weight tensor.")

    kernels = weight[output_channel].reshape(weight.shape[1], -1).double()
    if kernels.size(0) == 0:
        raise ValueError("Selected layer has no input channels.")
    return pairwise_correlation(kernels)


def pairwise_correlation(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError("pairwise_correlation expects a 2D tensor of shape (N, features).")
    count = samples.size(0)
    if count == 0:
        raise ValueError("Cannot compute correlation on an empty tensor.")
    if count == 1:
        return torch.ones((1, 1), dtype=torch.double, device=samples.device)

    samples = samples.double()
    centered = samples - samples.mean(dim=1, keepdim=True)
    numerator = centered @ centered.T
    norms = torch.linalg.norm(centered, dim=1, keepdim=True)
    denom = norms @ norms.T
    eps = 1e-12
    corr = torch.zeros_like(numerator)
    mask = denom > eps
    corr[mask] = numerator[mask] / denom[mask]
    corr.fill_diagonal_(1.0)
    return torch.clamp(corr, -1.0, 1.0)


def compute_channelwise_correlation(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 4:
        raise ValueError("Expected a 4D convolutional weight tensor.")
    channel_kernels = weight.sum(dim=1)  # (out_channels, kh, kw)
    flattened = channel_kernels.reshape(weight.shape[0], -1)
    return pairwise_correlation(flattened)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "model"


def _looks_like_directory(path: str) -> bool:
    if not path:
        return False
    if path.endswith(("/", "\\")):
        return True
    if os.path.isdir(path):
        return True
    _, ext = os.path.splitext(path)
    return ext == ""


def describe_model(ckpt_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    model_name = (
        str(metadata.get("model"))
        if metadata.get("model")
        else os.path.splitext(os.path.basename(ckpt_path))[0]
    )
    slug = _slugify(model_name)
    contrastive = bool(metadata.get("contrastive", False))
    display = f"[{'CKL' if contrastive else 'STD'}] {model_name}"
    return {
        "name": model_name,
        "slug": slug,
        "contrastive": contrastive,
        "display": display,
    }


def print_correlation_stats(label: str, matrix: torch.Tensor) -> None:
    print(
        f"{label}: correlation shape={tuple(matrix.shape)}, "
        f"min={matrix.min().item():.3f}, max={matrix.max().item():.3f}"
    )


def print_histogram_summary(label: str, histogram: torch.Tensor, bin_edges: List[float]) -> None:
    total_pairs = int(histogram.sum().item())
    print(f"{label}: total channel kernel pairs counted: {total_pairs}")
    ranges = [f"{bin_edges[i]:.1f} to {bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]
    for bucket, count in zip(ranges, histogram.tolist()):
        print(f"  {bucket:>12}: {count}")


def resolve_output_path(
    save_arg: Optional[str],
    ckpt_path: str,
    model_slug: str,
    layer_index: Optional[int],
    kernel_index: Optional[int],
    contrastive_enabled: bool,
    suffix: Optional[str] = None,
) -> str:
    prefix = "ckl-" if contrastive_enabled else ""
    base = f"{prefix}{model_slug}"
    if layer_index is not None:
        base += f"-layer{layer_index}"
    if kernel_index is not None:
        base += f"-kernel{kernel_index}"
    suffix_slug = _slugify(str(suffix)) if suffix else ""
    if suffix_slug:
        base += f"-{suffix_slug}"
    filename = f"{base}.png"

    if save_arg:
        save_arg = save_arg.strip()
        if _looks_like_directory(save_arg):
            return os.path.join(save_arg, filename)
        root, ext = os.path.splitext(save_arg)
        if ext:
            if suffix_slug:
                return f"{root}-{suffix_slug}{ext}"
            return save_arg
        return os.path.join(save_arg, filename)

    return filename


def format_colorbar_ticks(cbar) -> None:
    ticks = list(cbar.get_ticks())
    formatted = []
    for value in ticks:
        val = float(value)
        if abs(val) < 1e-8:
            formatted.append("0")
            continue
        label = f"{val:.2f}".rstrip("0").rstrip(".")
        formatted.append(label)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(formatted)


def get_plotting_modules(show: bool):
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - informative fallback
        raise RuntimeError("matplotlib is required for visualisation. Install it with 'pip install matplotlib'.") from exc

    if not show:
        matplotlib.use("Agg")  # type: ignore[attr-defined]

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - informative fallback
        raise RuntimeError("Failed to import matplotlib.pyplot.") from exc

    try:
        import seaborn as sns  # type: ignore
    except ImportError:
        sns = None
    else:
        sns.set_theme(context="talk", style="white")

    return plt, sns


def plot_correlation(matrix: torch.Tensor, title: str, save_path: Optional[str], show: bool) -> Optional[str]:
    plt, sns = get_plotting_modules(show)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor("white")

    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.detach().cpu().numpy()
    else:
        matrix_np = matrix

    if sns is not None:
        cmap = sns.color_palette("rocket", as_cmap=True)
    else:
        try:
            cmap = plt.get_cmap("viridis")
        except Exception:
            cmap = "viridis"  # Fallback string if custom cmap cannot be created

    if sns is not None:
        heatmap = sns.heatmap(
            matrix_np,
            ax=ax,
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            linewidths=0.0,
            cbar=True,
            cbar_kws={"shrink": 0.8, "label": "Correlation", "drawedges": False},
            xticklabels=False,
            yticklabels=False,
        )
        cbar = heatmap.collections[0].colorbar
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=9, length=0, width=0)
        format_colorbar_ticks(cbar)
        ax.tick_params(axis="both", bottom=False, left=False)
    else:
        cax = ax.imshow(matrix_np, cmap=cmap, vmin=-1.0, vmax=1.0)
        num_kernels = matrix_np.shape[0]
        ticks = list(range(num_kernels))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", length=0)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_visible(False)
        cbar.set_label("Correlation")
        cbar.ax.tick_params(length=0, width=0)
        format_colorbar_ticks(cbar)

    ax.set_xlabel("Input kernel index", fontsize=11, labelpad=8)
    ax.set_ylabel("Input kernel index", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f2f2f2")

    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def compute_similarity_histogram(
    correlation: torch.Tensor, bin_edges: List[float]
) -> torch.Tensor:
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("Similarity histogram expects a square correlation matrix.")

    channels = correlation.shape[0]
    if channels <= 1:
        return torch.zeros(len(bin_edges) - 1, dtype=torch.long)

    corr = correlation.clone().double()
    corr = torch.clamp(corr, -1.0, 1.0)
    triu_idx = torch.triu_indices(channels, channels, offset=1)
    if triu_idx.numel() == 0:
        return torch.zeros(len(bin_edges) - 1, dtype=torch.long)

    values = corr[triu_idx[0], triu_idx[1]]
    bins_tensor = torch.tensor(bin_edges, dtype=torch.double, device=values.device)
    indices = torch.bucketize(values, bins_tensor, right=False) - 1
    max_index = len(bin_edges) - 2
    indices = torch.clamp(indices, 0, max_index)

    counts = torch.zeros(len(bin_edges) - 1, dtype=torch.long, device=values.device)
    ones = torch.ones_like(indices, dtype=torch.long)
    counts.scatter_add_(0, indices.to(torch.long), ones)
    return counts.cpu()


def plot_similarity_distribution(
    counts: torch.Tensor,
    bin_edges: List[float],
    title: str,
    save_path: Optional[str],
    show: bool,
) -> Optional[str]:
    plt, sns = get_plotting_modules(show)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    fig.patch.set_facecolor("white")

    counts_np = counts.detach().cpu().numpy() if isinstance(counts, torch.Tensor) else counts
    bin_labels = [f"{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]
    positions = list(range(len(counts_np)))

    if sns is not None:
        sns.barplot(x=positions, y=counts_np, ax=ax, palette="rocket")
    else:
        cmap = plt.get_cmap("viridis")
        colors = [cmap(0.6) for _ in positions]
        ax.bar(positions, counts_np, color=colors)

    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel("Correlation range", fontsize=11, labelpad=8)
    ax.set_ylabel("Kernel pair count", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_facecolor("#f2f2f2")

    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def aggregate_layer_histograms(
    conv_layers: List[Tuple[str, torch.Tensor]],
    label: str,
    bin_edges: List[float],
) -> torch.Tensor:
    total = torch.zeros(len(bin_edges) - 1, dtype=torch.long)
    for idx, (name, weight) in enumerate(conv_layers):
        correlation = compute_channelwise_correlation(weight).cpu()
        layer_label = f"{label} layer[{idx}] {name}"
        print_correlation_stats(layer_label, correlation)
        histogram = compute_similarity_histogram(correlation, bin_edges)
        print_histogram_summary(layer_label, histogram, bin_edges)
        total += histogram
    print_histogram_summary(f"{label} (aggregate)", total, bin_edges)
    return total


def plot_channel_comparison(
    corr_a: torch.Tensor,
    corr_b: torch.Tensor,
    labels: Tuple[str, str],
    title: str,
    save_path: Optional[str],
    show: bool,
) -> Optional[str]:
    plt, sns = get_plotting_modules(show)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    fig.patch.set_facecolor("white")

    matrix_a = corr_a.detach().cpu().numpy()
    matrix_b = corr_b.detach().cpu().numpy()
    diff = matrix_a - matrix_b

    cmap_main = (
        sns.color_palette("rocket", as_cmap=True) if sns is not None else plt.get_cmap("viridis")
    )
    cmap_diff = (
        sns.diverging_palette(10, 240, as_cmap=True) if sns is not None else "coolwarm"
    )

    limit = float(max(abs(diff.min()), abs(diff.max()), 2.0))

    def _render(ax, matrix, cmap, vmin, vmax, title_text, cbar_label):
        ticks = (
            np.linspace(vmin, vmax, num=5)
            if abs(vmax - vmin) > 1e-8
            else np.array([vmin], dtype=float)
        )
        if sns is not None:
            heat = sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                square=True,
                linewidths=0.0,
                cbar=True,
                cbar_kws={"shrink": 0.8, "label": cbar_label, "drawedges": False},
                xticklabels=False,
                yticklabels=False,
            )
            cbar = heat.collections[0].colorbar
            cbar.outline.set_visible(False)
            cbar.set_ticks(ticks)
            format_colorbar_ticks(cbar)
        else:
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.outline.set_visible(False)
            cbar.set_label(cbar_label)
            cbar.set_ticks(ticks)
            format_colorbar_ticks(cbar)
        ax.set_title(title_text, fontsize=12, pad=10)
        ax.tick_params(axis="both", length=0)
        ax.set_facecolor("#f2f2f2")

    _render(axes[0], matrix_a, cmap_main, -1.0, 1.0, labels[0], "Correlation")
    _render(axes[1], matrix_b, cmap_main, -1.0, 1.0, labels[1], "Correlation")
    _render(
        axes[2],
        diff,
        cmap_diff,
        -limit,
        limit,
        f"Δ ({labels[0]} − {labels[1]})",
        "Difference",
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_similarity_distribution_comparison(
    counts_a: torch.Tensor,
    counts_b: torch.Tensor,
    bin_edges: List[float],
    labels: Tuple[str, str],
    title: str,
    save_path: Optional[str],
    show: bool,
) -> Optional[str]:
    plt, sns = get_plotting_modules(show)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.patch.set_facecolor("white")

    counts_a_np = counts_a.detach().cpu().numpy()
    counts_b_np = counts_b.detach().cpu().numpy()
    positions = list(range(len(bin_edges) - 1))
    bin_labels = [f"{edge:.1f}" for edge in bin_edges[1:]]
    width = 0.38

    if sns is not None:
        palette = sns.color_palette("rocket", 2)
    else:
        cmap = plt.get_cmap("viridis")
        palette = [cmap(0.2), cmap(0.7)]

    ax.bar(
        [p - width / 2 for p in positions],
        counts_a_np,
        width=width,
        color=palette[0],
        label=labels[0],
    )
    ax.bar(
        [p + width / 2 for p in positions],
        counts_b_np,
        width=width,
        color=palette[1],
        label=labels[1],
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel("Correlation upper bound", fontsize=11, labelpad=8)
    ax.set_ylabel("Kernel pair count", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_facecolor("#f2f2f2")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    try:
        state_dict, ckpt_meta = load_state_dict(args.ckpt)
    except (TypeError, RuntimeError, KeyError) as exc:
        raise SystemExit(str(exc))

    conv_layers = collect_conv_layers(state_dict)
    if not conv_layers:
        raise SystemExit("No convolutional layers with 4D weight tensors were found in the checkpoint.")

    primary_info = describe_model(args.ckpt, ckpt_meta)
    primary_info.update({"ckpt": args.ckpt, "conv_layers": conv_layers})

    compare_info: Optional[Dict[str, Any]] = None
    if args.compare_ckpt:
        if not os.path.isfile(args.compare_ckpt):
            raise SystemExit(f"Comparison checkpoint not found: {args.compare_ckpt}")
        try:
            compare_state, compare_meta = load_state_dict(args.compare_ckpt)
        except (TypeError, RuntimeError, KeyError) as exc:
            raise SystemExit(str(exc))
        compare_layers = collect_conv_layers(compare_state)
        if not compare_layers:
            raise SystemExit("Comparison checkpoint does not contain convolutional layers.")
        compare_info = describe_model(args.compare_ckpt, compare_meta)
        compare_info.update({"ckpt": args.compare_ckpt, "conv_layers": compare_layers})

    if args.list_layers:
        print("Primary checkpoint convolutional layers:\n")
        print_conv_layers(primary_info["conv_layers"])
        if compare_info:
            print("\nComparison checkpoint convolutional layers:\n")
            print_conv_layers(compare_info["conv_layers"])
        return

    mode = args.mode

    if mode == "all-layer":
        bin_edges = [round(-1.0 + 0.1 * i, 1) for i in range(21)]
        histogram_primary = aggregate_layer_histograms(
            primary_info["conv_layers"], primary_info["display"], bin_edges
        )

        if compare_info:
            histogram_compare = aggregate_layer_histograms(
                compare_info["conv_layers"], compare_info["display"], bin_edges
            )

            dist_compare_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index=None,
                kernel_index=None,
                contrastive_enabled=primary_info["contrastive"],
                suffix=f"all-layers-dist-compare-{compare_info['slug']}",
            )
            dist_title = "All layers · Kernel similarity distribution"
            dist_saved = plot_similarity_distribution_comparison(
                histogram_primary,
                histogram_compare,
                bin_edges,
                (primary_info["display"], compare_info["display"]),
                dist_title,
                dist_compare_path,
                args.show,
            )
            if dist_saved:
                print(f"All-layer distribution comparison saved to: {os.path.abspath(dist_saved)}")
        else:
            dist_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index=None,
                kernel_index=None,
                contrastive_enabled=primary_info["contrastive"],
                suffix="all-layers-dist",
            )
            dist_title = "All layers · Kernel similarity distribution"
            dist_saved = plot_similarity_distribution(
                histogram_primary,
                bin_edges,
                dist_title,
                dist_path,
                args.show,
            )
            if dist_saved:
                print(f"All-layer distribution saved to: {os.path.abspath(dist_saved)}")
        return

    if args.layer_index is None:
        raise SystemExit("--layer-index is required for this visualisation mode.")

    try:
        layer_index_primary = normalise_index(
            args.layer_index, len(primary_info["conv_layers"]), "Layer index"
        )
    except IndexError as exc:
        raise SystemExit(str(exc))

    layer_index_compare: Optional[int] = None
    if compare_info:
        try:
            layer_index_compare = normalise_index(
                args.layer_index,
                len(compare_info["conv_layers"]),
                "Layer index (comparison checkpoint)",
            )
        except IndexError as exc:
            raise SystemExit(str(exc))

    layer_name_primary, weight_primary = primary_info["conv_layers"][layer_index_primary]
    out_channels, in_channels, kernel_height, kernel_width = weight_primary.shape

    print(
        f"{primary_info['display']}: layer [{layer_index_primary}] {layer_name_primary} -> "
        f"shape={tuple(weight_primary.shape)}"
    )

    weight_compare: Optional[torch.Tensor] = None
    layer_name_compare: Optional[str] = None
    if compare_info and layer_index_compare is not None:
        layer_name_compare, weight_compare = compare_info["conv_layers"][layer_index_compare]
        print(
            f"{compare_info['display']}: layer [{layer_index_compare}] {layer_name_compare} -> "
            f"shape={tuple(weight_compare.shape)}"
        )
        if weight_compare.shape != weight_primary.shape:
            raise SystemExit(
                "Selected layers have different weight shapes; comparison is not supported."
            )

    if mode == "channel":
        if args.output_channel is None:
            raise SystemExit("--output-channel is required when --mode channel is selected.")
        try:
            output_channel_primary = normalise_index(
                args.output_channel, out_channels, "Output channel index"
            )
        except IndexError as exc:
            raise SystemExit(str(exc))

        correlation_primary = compute_kernel_correlation(weight_primary, output_channel_primary).cpu()
        print_correlation_stats(primary_info["display"], correlation_primary)

        if compare_info and weight_compare is not None:
            try:
                output_channel_compare = normalise_index(
                    args.output_channel,
                    weight_compare.shape[0],
                    "Output channel index (comparison checkpoint)",
                )
            except IndexError as exc:
                raise SystemExit(str(exc))

            correlation_compare = compute_kernel_correlation(
                weight_compare, output_channel_compare
            ).cpu()
            print_correlation_stats(compare_info["display"], correlation_compare)

            compare_suffix = f"compare-{compare_info['slug']}"
            compare_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index_primary,
                output_channel_primary,
                primary_info["contrastive"],
                suffix=compare_suffix,
            )

            title = f"Layer {layer_index_primary} · Output kernel {output_channel_primary}"
            comparison_saved = plot_channel_comparison(
                correlation_primary,
                correlation_compare,
                (primary_info["display"], compare_info["display"]),
                title,
                compare_path,
                args.show,
            )

            if comparison_saved:
                print(f"Comparison heatmap saved to: {os.path.abspath(comparison_saved)}")
        else:
            target_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index_primary,
                output_channel_primary,
                primary_info["contrastive"],
            )

            title = f"Layer {layer_index_primary} · Output kernel {output_channel_primary}"
            saved_path = plot_correlation(correlation_primary, title, target_path, args.show)

            if saved_path:
                print(f"Correlation heatmap saved to: {os.path.abspath(saved_path)}")

    elif mode == "all-channel":
        channel_correlation_primary = compute_channelwise_correlation(weight_primary).cpu()
        print_correlation_stats(primary_info["display"], channel_correlation_primary)

        bin_edges = [round(-1.0 + 0.1 * i, 1) for i in range(21)]
        histogram_primary = compute_similarity_histogram(channel_correlation_primary, bin_edges)
        print_histogram_summary(primary_info["display"], histogram_primary, bin_edges)

        if compare_info and weight_compare is not None:
            channel_correlation_compare = compute_channelwise_correlation(weight_compare).cpu()
            print_correlation_stats(compare_info["display"], channel_correlation_compare)

            histogram_compare = compute_similarity_histogram(channel_correlation_compare, bin_edges)
            print_histogram_summary(compare_info["display"], histogram_compare, bin_edges)

            corr_compare_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index_primary,
                kernel_index=None,
                contrastive_enabled=primary_info["contrastive"],
                suffix=f"channels-corr-compare-{compare_info['slug']}",
            )
            dist_compare_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index_primary,
                kernel_index=None,
                contrastive_enabled=primary_info["contrastive"],
                suffix=f"channels-dist-compare-{compare_info['slug']}",
            )

            corr_title = f"Layer {layer_index_primary} · Channel kernels"
            dist_title = f"Layer {layer_index_primary} · Kernel similarity distribution"

            corr_saved = plot_channel_comparison(
                channel_correlation_primary,
                channel_correlation_compare,
                (primary_info["display"], compare_info["display"]),
                corr_title,
                corr_compare_path,
                args.show,
            )
            dist_saved = plot_similarity_distribution_comparison(
                histogram_primary,
                histogram_compare,
                bin_edges,
                (primary_info["display"], compare_info["display"]),
                dist_title,
                dist_compare_path,
                args.show,
            )

            if corr_saved:
                print(f"Channel correlation comparison saved to: {os.path.abspath(corr_saved)}")
            if dist_saved:
                print(f"Similarity distribution comparison saved to: {os.path.abspath(dist_saved)}")
        else:
            corr_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index_primary,
                kernel_index=None,
                contrastive_enabled=primary_info["contrastive"],
                suffix="channels-corr",
            )
            dist_path = resolve_output_path(
                args.save,
                primary_info["ckpt"],
                primary_info["slug"],
                layer_index_primary,
                kernel_index=None,
                contrastive_enabled=primary_info["contrastive"],
                suffix="channels-dist",
            )

            corr_title = f"Layer {layer_index_primary} · Channel kernels"
            dist_title = f"Layer {layer_index_primary} · Kernel similarity distribution"

            corr_saved = plot_correlation(channel_correlation_primary, corr_title, corr_path, args.show)
            dist_saved = plot_similarity_distribution(
                histogram_primary, bin_edges, dist_title, dist_path, args.show
            )

            if corr_saved:
                print(f"Channel correlation heatmap saved to: {os.path.abspath(corr_saved)}")
            if dist_saved:
                print(f"Similarity distribution plot saved to: {os.path.abspath(dist_saved)}")
    else:
        raise SystemExit(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()

