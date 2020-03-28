import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



HIDDEN_UNITS = 15
LEARNING_RATE = 0.01  #0.001
EPOCH = 300      #400 ;
BATCH_SIZE = 32    #15
# Using 2 hidden layers  dnn网络
input_size = 6
num_classes = 4

class DNN(nn.Module):    #dnn网络
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 25)
        self.fc2 = nn.Linear(25, 10)
        self.fc3 = nn.Linear(10, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)
        # out = F.dropout(y_hat, p=0.2)
        return y_hat

def genarate_data(device):    #准备数据
    train = pd.read_csv('./car.data')
    test = pd.read_csv('./test.txt')

    # del  data['Unnamed: 13']
    # del data['OBJECTID']
    #print (data.shape[1])
    # data['危险性结果'] = data['危险性结果'].apply(chuliy)
    train['buying']=train['buying'].apply(chuliy1)
    train['maint'] = train['maint'].apply(chuliy1)
    train['doors'] = train['doors'].apply(chuliy4)
    train['persons'] = train['persons'].apply(chuliy3)
    train['lug_boot'] = train['lug_boot'].apply(chuliy6)
    train['safety'] = train['safety'].apply(chuliy5)
    train['feel'] = train['feel'].apply(chuliy2)

    test['buying'] = test['buying'].apply(chuliy1)
    test['maint'] = test['maint'].apply(chuliy1)
    test['doors'] = test['doors'].apply(chuliy4)
    test['persons'] = test['persons'].apply(chuliy3)
    test['lug_boot'] = test['lug_boot'].apply(chuliy6)
    test['safety'] = test['safety'].apply(chuliy5)
    test['feel'] = test['feel'].apply(chuliy2)

    # train, test = train_test_split(data, test_size=0.2, random_state=4)
    #print (test)

    all=train.append(test)
    all=pd.DataFrame(all)
    train, test = train_test_split(all, test_size=0.2, random_state=4)
    # print(all)
    train_label=train['feel'].values
    train_data=train.drop(['feel'],axis=1).values
    test_label = test['feel'].values
    test_data = test.drop(['feel'],axis=1).values
    test_data = torch.Tensor(test_data).to(device)
    test_label = torch.LongTensor(test_label).to(device)
    train_data = torch.Tensor(train_data).to(device)
    train_label = torch.LongTensor(train_label).to(device)
    return test_data, test_label, train_data, train_label,test

def main():
    # we want to use GPU if we have one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data, test_label, train_data, train_label,test = genarate_data(device)
    # prepare the data loader
    training_set = Data.TensorDataset(train_data,
                                      train_label)
    training_loader = Data.DataLoader(dataset=training_set,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    testing_set = Data.TensorDataset(test_data,
                                     test_label)
    testing_loader = Data.DataLoader(dataset=testing_set,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
    model = DNN(input_size, num_classes).to(device)
    # using crossentropy loss on classification problem
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH):
        correct_train = 0
        total_train = 0
        for (data, label) in training_loader:
            pred_label = model(data)
            loss = criterion(pred_label, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            _, answer = torch.max(pred_label.data, 1)

            total_train += label.size(0)
            correct_train += (answer == label).sum()
        print('Epoch {:3d} Accuracy on training data: {}% ({}/{})'
              .format(epoch, (100 * correct_train / total_train), correct_train, total_train))
        # pytorch 0.4 feature, not calculate grad on test set
        # 预测阶段，不跟新权值
        with torch.no_grad():
            correct_test = 0
            total_test = 0
            for (data, label) in testing_loader:
                pred_label = model(data)
                _, answer = torch.max(pred_label.data, 1)
                total_test += label.size(0)
                correct_test += (answer == label).sum()
            print('          Accuracy on testing data: {}% ({}/{})'
                  .format((100 * correct_test / total_test), correct_test, total_test))
    with torch.no_grad():
        predict = []
        label1 = []
        for (data, label) in testing_loader:
            pred_label = model(data)
            _, answer = torch.max(pred_label.data, 1)
            predict.append(answer.data.numpy().tolist())
            label1.append(label.data.numpy().tolist())
        predict = [i for j in predict for i in j]
        label1 = [i for j in label1 for i in j]
        ct = pd.DataFrame({'predict': predict, 'real': label1})
        test=test.reset_index(drop=True)
        final=pd.concat([test,ct],axis=1)
        final.to_csv("final.csv", index=False,encoding = "GB2312")
def chuliy1(data):
    if(data=='vhigh'):
        return 0
    elif(data=='high'):
        return 1
    elif(data=='med'):
        return 2
    elif(data=='low'):
        return 3
def chuliy2(data):
    if(data=='unacc'):
        return 0
    elif(data=='acc'):
        return 1
    elif(data=='vgood'):
        return 2
    elif(data=='good'):
        return 3
def chuliy3(data):
    if(data=='more'):
        return 0
    elif(data=='2'):
        return 1
    else:
        return 2
def chuliy4(data):
    if (data == '5more'):
        return 0
    elif (data == '2'):
        return 1
    elif (data=='3'):
        return 2
    else:
        return 3
def chuliy5(data):
    if(data=='low'):
        return 0
    elif(data=='med'):
        return 1
    elif(data=='high'):
        return 2
def chuliy6(data):
    if (data == 'small'):
        return 0
    elif (data == 'med'):
        return 1
    elif (data == 'big'):
        return 2
if __name__ == '__main__':
    main()



