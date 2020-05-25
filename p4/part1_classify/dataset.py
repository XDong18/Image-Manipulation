import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler


transform_val = transforms.Compose([transforms.ToTensor()])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.RandomVerticalFlip(),
    # transforms.RandomResizedCrop(28),
    transforms.ToTensor(),
    # transforms.RandomErasing()
])
    
train_set = torchvision.datasets.FashionMNIST(root='/shared/xudongliu/cs194', train=True,
                                        download=True, transform=transform_train)
val_set = torchvision.datasets.FashionMNIST(root='/shared/xudongliu/cs194', train=True,
                                        download=True, transform=transform_val)
test_set = torchvision.datasets.FashionMNIST(root='/shared/xudongliu/cs194', train=False,
                                        download=True, transform=transform_val)                                       
torch.manual_seed(0)
train_set_no, val_set_no = torch.utils.data.random_split(train_set, [50000, 10000])
train_sampler = SubsetRandomSampler(train_set_no.indices)
val_sampler = SubsetRandomSampler(val_set_no.indices)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat' ,'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

if __name__=='__main__':
    print(train_sampler)


