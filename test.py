import yaml
import argparse
import os
import pytorch_lightning as pl
from train import parse_args, Model, DataModule
import torch
torch.set_float32_matmul_precision('high')
def parse_args():
    parser = argparse.ArgumentParser(description="Train model with PyTorch Lightning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha parameter")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--margin", type=float, default=8, help="Margin for contrastive loss")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--mode", type=str, default="full-layer", help="full-layer or random-sampling")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist", help="Dataset to use")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every n epochs")
    parser.add_argument("--contrastive_kernel_loss", action="store_true", help="Use contrastive kernel loss")
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")

    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    
    return args

def main():
    args = parse_args()

    model = Model.load_from_checkpoint(
        checkpoint_path=args.resume,
        args=args,
        map_location=torch.device("cpu"),
    )
    model.to("cuda")
    model.eval()

    dm = DataModule(args)
    dm.prepare_data()
    dm.setup(stage="test")   # will populate dm.test_dataset

    if args.batch_size is not None:
        dm.batch_size = args.batch_size

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,            # disable logging
        enable_progress_bar=True
    )

    test_results = trainer.test(model=model, datamodule=dm)

    print("▶︎ Final test results:")
    for res in test_results:
        print(res)

if __name__ == "__main__":
    main()
