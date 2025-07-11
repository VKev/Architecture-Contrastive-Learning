import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, List


class SimpleMLP(nn.Module):
    """
    Fully-connected Multi-Layer Perceptron that accepts an arbitrary list of
    hidden-layer sizes.

    Args:
        input_size (int):  Number of input features (e.g. 4 for the Iris data).
        hidden_sizes (Sequence[int]):  Sizes of hidden layers, e.g. [64, 128, 256].
        num_classes (int):  Number of output classes.
        dropout_rate (float):  Dropout probability applied after every hidden layer
                               (set 0.0 to disable).
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_sizes: Sequence[int] | None = None,
        num_classes: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [16, 8]  # sensible default

        self.input_size = input_size
        self.hidden_sizes: List[int] = list(hidden_sizes)
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # ------------------------------------------------------------------ #
        # Dynamically build the feed-forward network
        # ------------------------------------------------------------------ #
        layers: list[nn.Module] = []
        in_features = input_size

        for h in self.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, h),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))  # output layer
        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    # ---------------------------------------------------------------------- #
    # Weight initialisation
    # ---------------------------------------------------------------------- #
    def _initialize_weights(self) -> None:
        """Xavier/Glorot initialization for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    # ---------------------------------------------------------------------- #
    # Forward & helpers
    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_size) → (B, num_classes)
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_size) → (B,)
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_size) → (B, num_classes)
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# -------------------------------------------------------------------------- #
# Convenience factory
# -------------------------------------------------------------------------- #
def create_flexible_mlp(
    input_size: int = 4,
    hidden_sizes: Sequence[int] | None = None,
    num_classes: int = 3,
    dropout_rate: float = 0.2,
) -> SimpleMLP:
    return SimpleMLP(input_size, hidden_sizes, num_classes, dropout_rate)


# -------------------------------------------------------------------------- #
# Quick smoke test
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Example: three hidden layers of sizes 64 → 128 → 256
    model = SimpleMLP(hidden_sizes=[64, 128, 256], dropout_rate=0.3)
    print(model)

    dummy = torch.randn(32, 4)
    out = model(dummy)
    print("Output logits shape:", out.shape)

    preds = model.predict(dummy)
    print("Predictions shape:", preds.shape)

    probs = model.predict_proba(dummy)
    print("Probabilities shape:", probs.shape)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}, trainable: {trainable:,}")
