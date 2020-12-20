import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

all = pd.read_csv('ElectionData.csv')
train, test = train_test_split(all, test_size=0.2, random_state=4)
train_label = train['target'].values
train_data = train.drop(['target'], axis=1).values
test_label = test['target'].values
test_data = test.drop(['target'], axis=1).values

test_data = torch.Tensor(test_data).to("cpu")
test_label = torch.LongTensor(test_label).to("cpu")
train_data = torch.Tensor(train_data).to("cpu")
train_label = torch.LongTensor(train_label).to("cpu")

def trainloader(batch_size = 64):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./', train=True,
    #                                         download=False, transform=transform_train)
    
    return  torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

def testloader(batch_size = 64):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # testset = torchvision.datasets.CIFAR10(root='./', train=False,
    #                                        download=False, transform=transform_test)
    
    return torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)