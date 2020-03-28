import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



HIDDEN_UNITS = 15
LEARNING_RATE = 0.001  #0.001
EPOCH = 300      #400 ;
BATCH_SIZE = 32    #15
# Using 2 hidden layers  dnn网络
input_size = 5
num_classes = 2

class DNN(nn.Module):    #dnn网络
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 20)
        # self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)
        # out = F.dropout(y_hat, p=0.2)
        return y_hat

def genarate_data(device):    #准备数据
    train = pd.read_excel('./data.xlsx')
    test = pd.read_excel('./test.xlsx')

    train_label = train['target'].values
    test_label = test['target'].values
    train=train.drop(['target'],axis=1)
    test=test.drop(['target'],axis=1)

    #85%

    #83%
    # del train['occ']
    # del test['occ']
    #
    # del train['edcnum']
    # del test['edcnum']
    #
    # del train['age']
    # del test['age']


    # 83%
    del train['wc']
    del train['nc']
    del train['fn']
    del train['race']
    del train['cl']
    del train['cg']

    del test['wc']
    del test['nc']
    del test['fn']
    del test['race']
    del test['cl']
    del test['cg']





    # 82.4%
    pca = PCA(n_components=5)
    train = pca.fit_transform(train)
    ratio_pca = pca.explained_variance_ratio_
    print(pca.explained_variance_ratio_)
    t_a=0
    for a in ratio_pca:
        t_a+=a
    print(t_a)
    # 0.9333174096067566
    test=pca.transform(test)

    # train=train.values
    # test=test.values


    # del  data['Unnamed: 13']
    # del data['OBJECTID']
    #print (data.shape[1])
    # data['危险性结果'] = data['危险性结果'].apply(chuliy)


    # train, test = train_test_split(data, test_size=0.2, random_state=4)
    #print (test)


    # print(all)

    # train_data=train.drop(['target'],axis=1).values
    train_data=train

    # test_data = test.drop(['target'],axis=1).values
    test_data=test
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
              .format(epoch,  ('%.2f' % (100 * float(correct_train) / float(total_train))), correct_train, total_train))
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
                  .format(('%.2f' % (100 * float(correct_test) / float(total_test))), correct_test, total_test))
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


if __name__ == '__main__':
    main()



