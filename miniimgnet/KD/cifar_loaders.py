import torchvision.transforms as transforms
import torchvision
import torch
from cifar_generator import CIFAR10

class CifarLoaders:
    def __init__(self):
        transform = transforms.Compose(
                    [transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        train_set_lg = CIFAR10(root='/mnt/hdd/cifar-10-python', train=True,
                                        download=True, transform=transform)
        self.train_loader_lg = torch.utils.data.DataLoader(train_set_lg, 
                                    batch_size=32, 
                                    shuffle=True, 
                                    num_workers=1)

        test_set_lg = CIFAR10(root='/mnt/hdd/cifar-10-python', train=False,
                                        download=True, transform=transform)
        self.test_loader_lg = torch.utils.data.DataLoader(test_set_lg, 
                                    batch_size=32, 
                                    shuffle=True, 
                                    num_workers=1)

        transform = transforms.Compose(
                    [
                        transforms.Resize((84, 84)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
        train_set = CIFAR10(root='/mnt/hdd/cifar-10-python', train=True,
                                        download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, 
                                    batch_size=32, 
                                    shuffle=True, 
                                    num_workers=1)
        test_set = CIFAR10(root='/mnt/hdd/cifar-10-python', train=False,
                                        download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, 
                                    batch_size=32, 
                                    shuffle=True, 
                                    num_workers=1)
    