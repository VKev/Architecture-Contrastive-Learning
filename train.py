import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms, models
import yaml
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
try:
    from torchmetrics import Accuracy
except ImportError:
    from pytorch_lightning.metrics import Accuracy
import warnings
import shutil
import wandb
import torch.nn.functional as F
import math
from typing import List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

torch.set_float32_matmul_precision('high')

# Import cifar10_models for VGG
try:
    from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
    CIFAR10_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: cifar10_models not available. VGG models will use default implementation.")
    CIFAR10_MODELS_AVAILABLE = False

class WarmupCosineLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]

from util import (
    transform_mnist,
    transform_mnist_resnet_train,
    transform_mnist_resnet_test,
    transform_cifar10_train,
    transform_cifar10_test,
    transform_cifar10_resnet_train,
    transform_cifar10_resnet_test,
    transform_mnist_224,
    transform_fashionmnist_train,
    transform_fashionmnist_test,
    transform_fashionmnist_resnet_train,
    transform_fashionmnist_resnet_test,
    transform_kmnist_train,
    transform_kmnist_test,
    transform_kmnist_resnet_train,
    transform_kmnist_resnet_test,
    transform_emnist_balanced_train,
    transform_emnist_balanced_test,
    transform_emnist_balanced_resnet_train,
    transform_emnist_balanced_resnet_test,
    transform_svhn_train,
    transform_svhn_test,
    ContrastiveKernelLoss,
    transform_imagenet_train,
    transform_imagenet_val,
    select_random_kernels,
    select_fixed_kernels,
    get_kernel_list,
)
from util.loss_neural import ContrastiveLinearLoss
from util.selective_neural import get_neuron_list, select_random_neurons, select_fixed_neurons
from model import ResNet50, LeNet5
from model.mlp import SimpleMLP
from model.resnet_cifar import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from model.googlenet import googlenet

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    print(f"Setting random seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    pl.seed_everything(seed, workers=True)


def _dataset_targets_array(dataset) -> np.ndarray:
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            data = getattr(dataset, attr)
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            return np.array(data)
    raise AttributeError("Dataset does not expose 'targets' or 'labels'.")


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.num_workers = 0  # Use 0 workers to avoid multiprocessing issues on Windows

    def prepare_data(self):
        """Download datasets that require it."""
        if self.dataset == "mnist":
            datasets.MNIST(root="./data", train=True, download=True)
            datasets.MNIST(root="./data", train=False, download=True)
        elif self.dataset == "fashion_mnist":
            datasets.FashionMNIST(root="./data", train=True, download=True)
            datasets.FashionMNIST(root="./data", train=False, download=True)
        elif self.dataset == "kmnist":
            datasets.KMNIST(root="./data", train=True, download=True)
            datasets.KMNIST(root="./data", train=False, download=True)
        elif self.dataset == "emnist_balanced":
            datasets.EMNIST(root="./data", split="balanced", train=True, download=True)
            datasets.EMNIST(root="./data", split="balanced", train=False, download=True)
        elif self.dataset == "cifar10":
            datasets.CIFAR10(root="./data", train=True, download=True)
            datasets.CIFAR10(root="./data", train=False, download=True)
        elif self.dataset == "cifar100":
            datasets.CIFAR100(root="./data", train=True, download=True)
            datasets.CIFAR100(root="./data", train=False, download=True)
        elif self.dataset == "svhn":
            datasets.SVHN(root="./data", split="train", download=True)
            datasets.SVHN(root="./data", split="test", download=True)
        elif self.dataset in ["iris", "breast_cancer", "imagenet1k"]:
            # These datasets are handled elsewhere or do not require downloads here
            pass

    def setup(self, stage=None):
        split = 0.9
        model_name = self.args.model.lower()
        cifar_resnet_models = {"resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202"}
        is_cifar_resnet = model_name in cifar_resnet_models

        if self.dataset == "mnist":
            train_transform = (
                transform_mnist_resnet_train
                if is_cifar_resnet
                else (
                    transform_mnist_224
                    if model_name in ["resnet50", "vgg16", "googlenet"]
                    else transform_mnist
                )
            )
            test_transform = (
                transform_mnist_resnet_test
                if is_cifar_resnet
                else (
                    transform_mnist_224
                    if model_name in ["resnet50", "vgg16", "googlenet"]
                    else transform_mnist
                )
            )

            full_dataset = datasets.MNIST(
                root="./data", train=True, transform=train_transform
            )
            self.test_dataset = datasets.MNIST(
                root="./data", train=False, transform=test_transform
            )
            
            # For MNIST, use all training data (no validation split)
            self.train_dataset = full_dataset
            self.val_dataset = self.test_dataset  # Use test set for validation monitoring
            return

        elif self.dataset == "cifar10":
            # Choose transforms based on model type
            if self.args.model.lower().startswith('resnet'):
                train_transform = transform_cifar10_resnet_train
                test_transform = transform_cifar10_resnet_test
            else:  # Default for GoogleNet and other models
                train_transform = transform_cifar10_train
                test_transform = transform_cifar10_test
                
            full_dataset = datasets.CIFAR10(
                root="./data", train=True, transform=train_transform
            )
            self.test_dataset = datasets.CIFAR10(
                root="./data", train=False, transform=test_transform
            )
            
            # For GoogleNet with CIFAR10, use all training data (no validation split)
            if self.args.model.lower() == "googlenet":
                self.train_dataset = full_dataset
                self.val_dataset = self.test_dataset  # Use test set for validation monitoring
                return

        elif self.dataset == "cifar100":
            # Choose transforms based on model type
            if self.args.model.lower().startswith('resnet'):
                train_transform = transform_cifar10_resnet_train
                test_transform = transform_cifar10_resnet_test
            else:  # Default for GoogleNet and other models
                train_transform = transform_cifar10_train
                test_transform = transform_cifar10_test
                
            full_dataset = datasets.CIFAR100(
                root="./data",
                train=True,
                transform=train_transform,
                download=True,
            )
            self.test_dataset = datasets.CIFAR100(
                root="./data",
                train=False,
                transform=test_transform,
                download=True,
            )

        elif self.dataset == "imagenet1k":
            full_dataset = datasets.ImageNet(
                root=r"D:\AI\Dataset\ImageNet1k",
                split="train",
                transform=transform_imagenet_train,
            )
            self.test_dataset = datasets.ImageNet(
                root=r"D:\AI\Dataset\ImageNet1k",
                split="val",
                transform=transform_imagenet_val,
            )
            split = 0.98

        elif self.dataset == "iris":
            # Load Iris dataset from sklearn
            iris = load_iris()
            X, y = iris.data, iris.target
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=42
            )
            
            # Create TensorDatasets
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_test, y_test)  # Use test set for validation
            self.test_dataset = TensorDataset(X_test, y_test)
            
            return  # Skip the general dataset splitting logic below

        elif self.dataset == "breast_cancer":
            # Load Breast Cancer dataset from sklearn
            cancer = load_breast_cancer()
            X, y = cancer.data, cancer.target
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            
            # Split data: 50% train, 50% test (following user's preference)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, stratify=y, random_state=42
            )
            
            # Create TensorDatasets
            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_test, y_test)  # Use test set for validation
            self.test_dataset = TensorDataset(X_test, y_test)
            
            return  # Skip the general dataset splitting logic below

        if self.dataset not in ["iris", "breast_cancer"]:
            labels = _dataset_targets_array(full_dataset)
            train_idx, val_idx = train_test_split(
                np.arange(len(full_dataset)),
                test_size=1.0 - split,
                stratify=labels,
                random_state=42,
            )

            self.train_dataset = Subset(full_dataset, train_idx)
            self.val_dataset = Subset(full_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        common_kwargs = dict(
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        # For GoogleNet with CIFAR10 or MNIST, only return test loader (no separate validation)
        if (self.args.model.lower() == "googlenet" and self.dataset == "cifar10") or self.dataset == "mnist":
            test_loader = DataLoader(self.test_dataset, **common_kwargs)
            return test_loader
        
        val_loader = DataLoader(self.val_dataset, **common_kwargs)
        test_loader = DataLoader(self.test_dataset, **common_kwargs)
        return [val_loader, test_loader]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        color_datasets = {"cifar10", "cifar100", "imagenet1k", "svhn"}
        channels = 3 if args.dataset in color_datasets else 1
        resnet_supported_datasets = {
            "cifar10",
            "cifar100",
            "mnist",
            "fashion_mnist",
            "kmnist",
            "emnist_balanced",
            "svhn",
        }
        resnet_supported_str = ", ".join(sorted(resnet_supported_datasets))
        self.num_classes = 10
        if args.dataset in ["imagenet1k"]:
            self.num_classes = 1000
        elif args.dataset in ["cifar100"]:
            self.num_classes = 100
        elif args.dataset in ["emnist_balanced"]:
            self.num_classes = 47
        elif args.dataset in ["iris"]:
            self.num_classes = 3
        elif args.dataset in ["breast_cancer"]:
            self.num_classes = 2

        # Initialize accuracy metric for CIFAR VGG models
        self.using_cifar_vgg = (
            args.model.lower().startswith("vgg")
            and args.dataset in ["cifar10", "cifar100"]
            and CIFAR10_MODELS_AVAILABLE
        )
        if self.using_cifar_vgg:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        if args.model.lower() == "simplemlp":
            if args.dataset == "iris":
                self.model = SimpleMLP(input_size=4, hidden_sizes=[64,128,256,64], num_classes=3, dropout_rate=0.1)
            elif args.dataset == "breast_cancer":
                self.model = SimpleMLP(input_size=30, hidden_sizes=[64,128,256,64], num_classes=2, dropout_rate=0.1)
            else:
                raise ValueError(f"{args.model} is designed for Iris and Breast Cancer datasets only")

        elif args.model.lower() == "resnet50":
            self.model = ResNet50(num_classes=self.num_classes, channels=channels)

        elif args.model.lower() == "vgg16_bn":
            if self.using_cifar_vgg:
                # Use custom CIFAR-optimized VGG16 and match class count
                self.model = vgg16_bn(num_classes=self.num_classes)
            else:
                # Use default torchvision VGG16 with batch normalization
                self.model = models.vgg16_bn(weights=None)
                if channels == 1:
                    self.model.features[0] = nn.Conv2d(
                        1, 64, kernel_size=3, stride=1, padding=1
                    )
                self.model.classifier[6] = nn.Linear(4096, self.num_classes)

        elif args.model.lower() == "vgg11_bn":
            if self.using_cifar_vgg:
                # Use custom CIFAR-optimized VGG11
                self.model = vgg11_bn(num_classes=self.num_classes)
            else:
                raise ValueError(f"{args.model} requires the CIFAR VGG definitions; install them or use torchvision VGG instead")

        elif args.model.lower() == "vgg13_bn":
            if self.using_cifar_vgg:
                # Use custom CIFAR-optimized VGG13
                self.model = vgg13_bn(num_classes=self.num_classes)
            else:
                raise ValueError(f"{args.model} requires the CIFAR VGG definitions; install them or use torchvision VGG instead")

        elif args.model.lower() == "vgg19_bn":
            if self.using_cifar_vgg:
                # Use custom CIFAR-optimized VGG19
                self.model = vgg19_bn(num_classes=self.num_classes)
            else:
                raise ValueError(f"{args.model} requires the CIFAR VGG definitions; install them or use torchvision VGG instead")

        elif args.model.lower() == "lenet5":
            if channels == 1:
                self.model = LeNet5()
            else:
                raise ValueError(f"{args.model} only supports 1-channel input")

        elif args.model.lower() == "googlenet":
            # Use custom CIFAR-optimized GoogleNet implementation
            self.model = googlenet()
            
            # Adjust final linear layer for correct number of classes
            # The custom GoogleNet has a linear layer that outputs 10 classes by default
            if self.num_classes != 10:
                in_features = self.model.linear.in_features
                self.model.linear = nn.Linear(in_features, self.num_classes)
            
            # Handle single channel input (for MNIST)
            if channels == 1:
                self.model.pre_layers[0] = nn.Conv2d(1, 192, kernel_size=3, padding=1)

        # CIFAR ResNet models
        elif args.model.lower() == "resnet20":
            if args.dataset in resnet_supported_datasets:
                self.model = resnet20()
                if self.num_classes != 10:  # Adjust for CIFAR100
                    self.model.linear = nn.Linear(64, self.num_classes)
            else:
                raise ValueError(f"{args.model} supports datasets: {resnet_supported_str}")

        elif args.model.lower() == "resnet32":
            if args.dataset in resnet_supported_datasets:
                self.model = resnet32()
                if self.num_classes != 10:  # Adjust for CIFAR100
                    self.model.linear = nn.Linear(64, self.num_classes)
            else:
                raise ValueError(f"{args.model} supports datasets: {resnet_supported_str}")

        elif args.model.lower() == "resnet44":
            if args.dataset in resnet_supported_datasets:
                self.model = resnet44()
                if self.num_classes != 10:  # Adjust for CIFAR100
                    self.model.linear = nn.Linear(64, self.num_classes)
            else:
                raise ValueError(f"{args.model} supports datasets: {resnet_supported_str}")

        elif args.model.lower() == "resnet56":
            if args.dataset in resnet_supported_datasets:
                self.model = resnet56()
                if self.num_classes != 10:  # Adjust for CIFAR100
                    self.model.linear = nn.Linear(64, self.num_classes)
            else:
                raise ValueError(f"{args.model} supports datasets: {resnet_supported_str}")

        elif args.model.lower() == "resnet110":
            if args.dataset in resnet_supported_datasets:
                self.model = resnet110()
                if self.num_classes != 10:  # Adjust for CIFAR100
                    self.model.linear = nn.Linear(64, self.num_classes)
            else:
                raise ValueError(f"{args.model} supports datasets: {resnet_supported_str}")

        elif args.model.lower() == "resnet1202":
            if args.dataset in resnet_supported_datasets:
                self.model = resnet1202()
                if self.num_classes != 10:  # Adjust for CIFAR100
                    self.model.linear = nn.Linear(64, self.num_classes)
            else:
                raise ValueError(f"{args.model} supports datasets: {resnet_supported_str}")

        else:
            raise ValueError(f"Unsupported model: {args.model}")

        self.cls_criterion = nn.CrossEntropyLoss()
        self.kernel_loss_fn = ContrastiveKernelLoss(margin=args.margin)
        self.linear_loss_fn = ContrastiveLinearLoss(margin=args.margin)

        print(self.model)
        
        print(f"Contrastive loss calculation: {'Enabled' if args.calculate_contrastive_loss else 'Disabled'}")
        
        if args.calculate_contrastive_loss:
            if args.model.lower() == "simplemlp":
                print("SimpleMLP model detected")
                print(f"Contrastive linear loss: {'Enabled' if args.contrastive_linear_loss else 'Disabled'}")
                if args.contrastive_linear_loss:
                    print(f"Linear loss mode: {args.mode}")
                    print(f"Number of neurons: {args.num_kernels}")
                    print(f"Margin: {args.margin}, Alpha: {args.alpha}")
            else:
                print(f"Channel diversity mode: {'Enabled' if args.channel_diversity else 'Disabled'}")
                print(f"Select layer mode: {args.select_layer_mode}")
                print(f"Kernel selection mode: {args.mode}")
                print(f"Number of kernels: {args.num_kernels}")
                print(f"Contrastive kernel loss: {'Enabled' if args.contrastive_kernel_loss else 'Disabled'}")
                if args.contrastive_kernel_loss:
                    print(f"Margin: {args.margin}, Alpha: {args.alpha}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.using_cifar_vgg:
            # Custom optimizer and scheduler for CIFAR VGG models
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=getattr(self.args, 'weight_decay', 5e-4),
                momentum=0.9,
                nesterov=True,
            )
            
            # Calculate total steps based on actual dataset size and batch size
            if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                # If datamodule is available, calculate from train dataset
                if hasattr(self.trainer.datamodule, 'train_dataset'):
                    train_dataset_size = len(self.trainer.datamodule.train_dataset)
                    steps_per_epoch = train_dataset_size // self.args.batch_size
                    total_steps = self.args.num_epochs * steps_per_epoch
                else:
                    # Fallback: CIFAR10 has 50,000 training samples
                    print("FALL BACK")
                    steps_per_epoch = 50000 // self.args.batch_size
                    total_steps = self.args.num_epochs * steps_per_epoch
            else:
                # Fallback: CIFAR10 has 50,000 training samples
                print("FALL BACK")
                steps_per_epoch = 50000 // self.args.batch_size
                total_steps = self.args.num_epochs * steps_per_epoch
            
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer, 
                    warmup_epochs=int(total_steps * 0.3), 
                    max_epochs=total_steps
                ),
                "interval": "step",
                "name": "learning_rate",
            }
            return [optimizer], [scheduler]
            
        elif self.args.model.lower() in ["simplemlp"]:
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            
        elif self.args.model.lower() in ["googlenet"]:
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.args.lr, 
                momentum=0.9, 
                weight_decay=self.args.weight_decay
            )
            
            # Custom scheduler with minimum learning rate of 0.001
            def lr_lambda(epoch):
                # Step decay every 50 epochs with gamma=0.1, but minimum lr is 0.001
                initial_lr = self.args.lr
                min_lr = 0.001
                
                # Calculate the number of steps
                steps = epoch // 50
                new_lr = initial_lr * (0.1 ** steps)
                
                # Return the ratio relative to initial_lr, but ensure it's not below min_lr/initial_lr
                return max(new_lr / initial_lr, min_lr / initial_lr)
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        elif self.args.model.lower() in [
            "resnet50",
            "resnet20",
            "resnet32", 
            "resnet44",
            "resnet56",
            "resnet110",
            "resnet1202",
        ]:
            optimizer = optim.SGD(
                self.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay
            )

            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[125, 125+31], gamma=0.1
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        model_output = self.model(x)

        # Handle model outputs (custom GoogleNet returns simple tensor)
        if hasattr(model_output, 'logits'):  # PyTorch GoogleNet with aux_logits=True (not used anymore)
            logits = model_output.logits
            aux_logits1 = getattr(model_output, 'aux_logits1', None)
            aux_logits2 = getattr(model_output, 'aux_logits2', None)
        else:  # Regular models including custom GoogleNet
            logits = model_output
            aux_logits1 = None
            aux_logits2 = None

        # 1) Classification loss
        cls_loss = self.cls_criterion(logits, y)
        
        # Add auxiliary losses for PyTorch GoogleNet if available (not used with custom GoogleNet)
        if aux_logits1 is not None:
            cls_loss += 0.3 * self.cls_criterion(aux_logits1, y)
        if aux_logits2 is not None:
            cls_loss += 0.3 * self.cls_criterion(aux_logits2, y)

        # 2) Calculate contrastive loss for monitoring (but may not use it in final loss)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        
        if self.hparams["calculate_contrastive_loss"]:
            if self.hparams["model"].lower() == "simplemlp":
                # Calculate contrastive linear loss for SimpleMLP
                neuron_list = get_neuron_list(self.model, select_layer_mode=self.hparams["select_layer_mode"])
                if self.hparams["mode"].lower() == "random-sampling":
                    neuron_list = select_random_neurons(neuron_list, k=self.hparams["num_kernels"])
                elif self.hparams["mode"].lower() == "fixed-sampling":
                    neuron_list = select_fixed_neurons(neuron_list, k=self.hparams["num_kernels"], seed=self.args.seed)
                contrastive_loss = (
                    self.linear_loss_fn(neuron_list)
                    if neuron_list
                    else torch.tensor(0.0, device=self.device)
                )
            else:
                # Calculate contrastive kernel loss for other models
                kernel_list = get_kernel_list(self.model, channel_diversity=self.hparams["channel_diversity"], select_layer_mode=self.hparams["select_layer_mode"])
                if self.hparams["mode"].lower() == "random-sampling":
                    kernel_list = select_random_kernels(kernel_list, k=self.hparams["num_kernels"])
                elif self.hparams["mode"].lower() == "fixed-sampling":
                    kernel_list = select_fixed_kernels(kernel_list, k=self.hparams["num_kernels"], seed=self.args.seed)
                contrastive_loss = (
                    self.kernel_loss_fn(kernel_list)
                    if kernel_list
                    else torch.tensor(0.0, device=self.device)
                )

        # 3) Total loss - only add contrastive loss if flag is enabled
        total_loss = cls_loss
        use_contrastive = (
            self.hparams["contrastive_linear_loss"] if self.hparams["model"].lower() == "simplemlp"
            else self.hparams["contrastive_kernel_loss"]
        )
        if use_contrastive:
            total_loss = total_loss + self.hparams["alpha"] * contrastive_loss

        # 4) Compute accuracy and log metrics
        if self.using_cifar_vgg:
            # Use torchmetrics accuracy for VGG on CIFAR10 (returns 0-1 range)
            acc = self.accuracy(logits, y)
        else:
            # Use manual calculation for other models
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=False)
        
        # Log appropriate contrastive loss
        if self.hparams["model"].lower() == "simplemlp":
            self.log("train/linear_loss", self.hparams["alpha"] * contrastive_loss, on_step=True, on_epoch=True, prog_bar=False)
        else:
            self.log("train/kernel_loss", self.hparams["alpha"] * contrastive_loss, on_step=True, on_epoch=True, prog_bar=False)

        optimizer = self.optimizers()  # returns a list, take the first optimizer
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False)

        return total_loss

    def on_after_backward(self):
        total_norm = torch.norm(
            torch.stack(
                [
                    p.grad.detach().norm(2)
                    for p in self.parameters()
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        self.log(
            "grad_norm/global", total_norm, on_step=True, on_epoch=False, prog_bar=False
        )

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().norm(2)
                self.log(
                    f"grad_norm/{name}",
                    grad_norm,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        model_output = self.model(x)

        # Handle model outputs (custom GoogleNet returns simple tensor)
        if hasattr(model_output, 'logits'):  # PyTorch GoogleNet with aux_logits=True (not used anymore)
            logits = model_output.logits
            aux_logits1 = getattr(model_output, 'aux_logits1', None)
            aux_logits2 = getattr(model_output, 'aux_logits2', None)
        else:  # Regular models including custom GoogleNet
            logits = model_output
            aux_logits1 = None
            aux_logits2 = None

        cls_loss = self.cls_criterion(logits, y)
        
        # Add auxiliary losses for PyTorch GoogleNet if available (not used with custom GoogleNet)
        if aux_logits1 is not None:
            cls_loss += 0.3 * self.cls_criterion(aux_logits1, y)
        if aux_logits2 is not None:
            cls_loss += 0.3 * self.cls_criterion(aux_logits2, y)
        
        # Calculate contrastive loss for monitoring (but may not use it in final loss)
        contrastive_loss = torch.tensor(0.0, device=self.device)
        
        if self.hparams["calculate_contrastive_loss"]:
            if self.hparams["model"].lower() == "simplemlp":
                # Calculate contrastive linear loss for SimpleMLP
                neuron_list = get_neuron_list(self.model, select_layer_mode=self.hparams["select_layer_mode"])
                if self.hparams["mode"].lower() == "random-sampling":
                    neuron_list = select_random_neurons(neuron_list, k=self.hparams["num_kernels"])
                elif self.hparams["mode"].lower() == "fixed-sampling":
                    neuron_list = select_fixed_neurons(neuron_list, k=self.hparams["num_kernels"], seed=self.args.seed)
                contrastive_loss = (
                    self.linear_loss_fn(neuron_list)
                    if neuron_list
                    else torch.tensor(0.0, device=self.device)
                )
            else:
                # Calculate contrastive kernel loss for other models
                kernel_list = get_kernel_list(self.model, channel_diversity=self.hparams["channel_diversity"], select_layer_mode=self.hparams["select_layer_mode"])
                if self.hparams["mode"].lower() == "random-sampling":
                    kernel_list = select_random_kernels(kernel_list, k=self.hparams["num_kernels"])
                elif self.hparams["mode"].lower() == "fixed-sampling":
                    kernel_list = select_fixed_kernels(kernel_list, k=self.hparams["num_kernels"], seed=self.args.seed)
                contrastive_loss = (
                    self.kernel_loss_fn(kernel_list)
                    if kernel_list
                    else torch.tensor(0.0, device=self.device)
                )

        # Total loss
        use_contrastive = (
            self.hparams["contrastive_linear_loss"] if self.hparams["model"].lower() == "simplemlp"
            else self.hparams["contrastive_kernel_loss"]
        )
        total_loss = cls_loss + (
            self.hparams["alpha"] * contrastive_loss if use_contrastive else 0.0
        )

        # Compute accuracy
        if self.using_cifar_vgg:
            # Use torchmetrics accuracy for VGG on CIFAR10 (returns 0-1 range)
            acc = self.accuracy(logits, y)
        else:
            # Use manual calculation for other models
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        # For GoogleNet with CIFAR10 or MNIST, only log as test (no separate validation)
        if (self.hparams["model"].lower() == "googlenet" and self.hparams["dataset"] == "cifar10") or self.hparams["dataset"] == "mnist":
            self.log("test/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False) 
            self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("test/cls_loss", cls_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            
            # Log appropriate contrastive loss
            if self.hparams["model"].lower() == "simplemlp":
                self.log("test/linear_loss", self.hparams["alpha"] * contrastive_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            else:
                self.log("test/kernel_loss", self.hparams["alpha"] * contrastive_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)  # For checkpoint monitoring
        
        elif dataloader_idx == 0:  # Validation
            self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("val/cls_loss", cls_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            
            # Log appropriate contrastive loss
            if self.hparams["model"].lower() == "simplemlp":
                self.log("val/linear_loss", self.hparams["alpha"] * contrastive_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            else:
                self.log("val/kernel_loss", self.hparams["alpha"] * contrastive_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        else:  # Test
            self.log("test/loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False) 
            self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log("test/cls_loss", cls_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            
            # Log appropriate contrastive loss
            if self.hparams["model"].lower() == "simplemlp":
                self.log("test/linear_loss", self.hparams["alpha"] * contrastive_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            else:
                self.log("test/kernel_loss", self.hparams["alpha"] * contrastive_loss, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)  # For checkpoint monitoring
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with PyTorch Lightning")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha parameter")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for contrastive loss")
    parser.add_argument("--num_kernels", type=float, default=128, help="Number of kernels for contrastive loss (int for exact count, float <1 for percentage)")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture (resnet50, vgg16, vgg11_bn, vgg13_bn, vgg19_bn, lenet5, googlenet, resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202, simplemlp)")
    parser.add_argument("--mode", type=str, default="full-layer", help="Kernel selection mode: full-layer, random-sampling, or fixed-sampling")
    parser.add_argument(
        "--dataset",
        choices=[
            "mnist",
            "fashion_mnist",
            "kmnist",
            "emnist_balanced",
            "cifar10",
            "cifar100",
            "svhn",
            "imagenet1k",
            "iris",
            "breast_cancer",
        ],
        default="mnist",
        help="Dataset to use",
    )
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every n epochs")
    parser.add_argument("--contrastive_kernel_loss", action="store_true", help="Use contrastive kernel loss")
    parser.add_argument("--contrastive_linear_loss", action="store_true", help="Use contrastive linear loss (only for SimpleMLP)")
    parser.add_argument("--calculate_contrastive_loss", action="store_true", help="Enable contrastive loss calculation (required for contrastive_kernel_loss or contrastive_linear_loss)")
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--channel_diversity", action="store_true", help="Use channel diversity mode for kernel extraction and loss calculation")
    parser.add_argument("--select_layer_mode", type=str, default="default", choices=["default", "filter"], help="Layer selection mode: 'default' uses all Conv2d layers, 'filter' uses alternating pattern (take, ignore, take, ignore...)")

    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
            print(f"Overriding {key}: {getattr(args, key)} -> {value}")

    return args


def build_model(args):
    print("Building model")
    if args.resume:
        print("Resuming from checkpoint:", args.resume)
        return Model.load_from_checkpoint(args.resume, args=args)
    return Model(args)


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Login to wandb only if it's being used
    if args.wandb:
        wandb.login(key="b8b74d6af92b4dea7706baeae8b86083dad5c941")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        hparams = checkpoint["hyper_parameters"]
        if "wandb_id" in hparams:
            args.wandb_id = hparams["wandb_id"]
        for key, value in hparams.items():
            if key != "resume" and hasattr(args, key) and getattr(args, key) != value:
                print(f"Overriding {key}: {getattr(args, key)} -> {value}")
                setattr(args, key, value)
        checkpoint = None
    else:
        args.wandb_id = wandb.util.generate_id()

    model = build_model(args)
    dm = DataModule(args)

    callbacks = []
    ckl_suffix = f"-ckl-n-{args.num_kernels}-m-{args.margin}-a-{args.alpha}" if args.contrastive_kernel_loss else ""
    cll_suffix = f"-cll-n-{args.num_kernels}-m-{args.margin}-a-{args.alpha}" if args.contrastive_linear_loss else ""
    cd_suffix = ("-cd" if args.channel_diversity else "" ) if args.contrastive_kernel_loss else ""
    
    # Choose appropriate suffix based on model type
    contrastive_suffix = cll_suffix if args.model.lower() == "simplemlp" else ckl_suffix
    
    ModelCheckpoint.CHECKPOINT_NAME_LAST = (
        f"{args.model}-{args.dataset}"
        + contrastive_suffix
        + cd_suffix
        + "-{epoch}-{test_acc:.4f}-last"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoint",
        filename=f"{args.model}-{args.dataset}"
        + contrastive_suffix
        + cd_suffix
        + "-{epoch}-{test_acc:.4f}",
        monitor="test_acc",
        mode="max",
        save_top_k=1,
    )
    callbacks.append(checkpoint_callback)

    if args.early_stopping:
        # For GoogleNet with CIFAR10 or MNIST, monitor test accuracy instead of validation
        if (args.model.lower() == "googlenet" and args.dataset == "cifar10") or args.dataset == "mnist":
            early_stopping = EarlyStopping(
                monitor="test/acc", patience=args.patience, mode="max", verbose=True
            )
        else:
            early_stopping = EarlyStopping(
                monitor="val/acc", patience=args.patience, mode="max", verbose=True
            )
        callbacks.append(early_stopping)

    logger = None
    if args.wandb:
        wandb_name = f"{args.model}-{args.dataset}" + contrastive_suffix + cd_suffix
        logger = WandbLogger(
            project="architecture-contrastive-loss",
            name=wandb_name,
            id=args.wandb_id,
            resume="allow",
            offline=False,
        )
    trainer_kwargs = dict(
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices=args.device,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
    )

    if torch.cuda.device_count() > 1:
        trainer_kwargs.update(
            strategy="ddp",
            sync_batchnorm=True,
            deterministic=True,
        )
    else:
        trainer_kwargs["deterministic"] = False

    trainer = pl.Trainer(**trainer_kwargs)

    shutil.rmtree("./wandb", ignore_errors=True)
    shutil.rmtree("./lightning_logs", ignore_errors=True)

    trainer.fit(
        model, datamodule=dm, ckpt_path=args.resume if args.resume else None
    )


if __name__ == "__main__":
    main()
