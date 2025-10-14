import torchvision.transforms as transforms

transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_mnist_224 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# GoogleNet CIFAR10 transforms (default/original normalization)
transform_cifar10_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),  
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ResNet CIFAR10 transforms (0.5, 0.5, 0.5 normalization)
transform_cifar10_resnet_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_cifar10_resnet_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_imagenet_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_imagenet_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _make_grayscale_train(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])


def _make_grayscale_test(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])


def _make_grayscale_resnet_train(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std)),
    ])


def _make_grayscale_resnet_test(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std)),
    ])


transform_mnist_resnet_train = _make_grayscale_resnet_train(0.1307, 0.3081)
transform_mnist_resnet_test = _make_grayscale_resnet_test(0.1307, 0.3081)

transform_fashionmnist_train = _make_grayscale_train(0.2860, 0.3530)
transform_fashionmnist_test = _make_grayscale_test(0.2860, 0.3530)
transform_fashionmnist_resnet_train = _make_grayscale_resnet_train(0.2860, 0.3530)
transform_fashionmnist_resnet_test = _make_grayscale_resnet_test(0.2860, 0.3530)

transform_kmnist_train = _make_grayscale_train(0.1918, 0.3483)
transform_kmnist_test = _make_grayscale_test(0.1918, 0.3483)
transform_kmnist_resnet_train = _make_grayscale_resnet_train(0.1918, 0.3483)
transform_kmnist_resnet_test = _make_grayscale_resnet_test(0.1918, 0.3483)

transform_emnist_balanced_train = _make_grayscale_train(0.1751, 0.3330)
transform_emnist_balanced_test = _make_grayscale_test(0.1751, 0.3330)
transform_emnist_balanced_resnet_train = _make_grayscale_resnet_train(0.1751, 0.3330)
transform_emnist_balanced_resnet_test = _make_grayscale_resnet_test(0.1751, 0.3330)

transform_svhn_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])

transform_svhn_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
])
