import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
#print(torch.__version__)
# hyperparameters
HIDDEN_UNITS = 1000
LEARNING_RATE = 0.001  #0.001
EPOCH = 400      #400 best
BATCH_SIZE = 70    #15
# Using 2 hidden layers  dnn网络
input_size = 8
num_classes = 4
class DNN(nn.Module):    #dnn网络
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 17)
        # self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(17, num_classes)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        y_hat = self.fc3(x)
        return y_hat
'''
class DNN(nn.Module):   #原来的网络
    def __init__(self):
        super(DNN, self).__init__()
        self.linear_1 = nn.Linear(12, HIDDEN_UNITS)
        self.linear_2 = nn.Linear(HIDDEN_UNITS, 4)
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        # using softmax to output the probabilities of each class
        x = F.softmax(self.linear_2(x), dim=0)
        return x
'''
def genarate_data(device):    #准备数据

    # 训练集：91.92%
    # 测试集：91.81%
    data = pd.read_excel('data.xlsx')
    del data['Unnamed: 13']
    del data['OBJECTID']
    # 训练集：90.29%
    # 测试集：90.23%
    del data['断裂距离']
    del data['距公路距离']
    del data['距水系距离']
    del data['距面状水距离']


    #print (data.shape[1])
    data['危险性结果'] = data['危险性结果'].apply(chuliy)
    train, test = train_test_split(data, test_size=0.2, random_state=4)
    #print (test)
    train_label=train['危险性结果'].values
    train_data=train.drop(['危险性结果'],axis=1).values
    test_label = test['危险性结果'].values
    test_data = test.drop(['危险性结果'],axis=1).values
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
              .format(epoch, ('%.2f' % (100 * float(correct_train) / float(total_train))), correct_train, total_train))
        # pytorch 0.4 feature, not calculate grad on test set
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
    # with torch.no_grad():
        # predict = []
        # label1 = []
        # for (data, label) in testing_loader:
        #     pred_label = model(data)
        #     _, answer = torch.max(pred_label.data, 1)
        #     predict.append(answer.data.numpy().tolist())
        #     label1.append(label.data.numpy().tolist())
        # predict = [i for j in predict for i in j]
        # label1 = [i for j in label1 for i in j]
        # ct = pd.DataFrame({'predict': predict, 'real': label1})
        # test=test.reset_index(drop=True)
        # final=pd.concat([test,ct],axis=1)
        # final.to_csv("final.csv", index=False,encoding = "GB2312")
def chuliy(t):
    if (t==1):
        return 0
    elif (t==2):
        return 1
    elif (t==4):
        return 2
    else :
        return 3
if __name__ == '__main__':
    main()





