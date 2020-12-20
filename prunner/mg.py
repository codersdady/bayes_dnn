import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.utils.prune as prune

from sklearn.decomposition import PCA

#print(torch.__version__)
# hyperparameters
HIDDEN_UNITS = 1000
LEARNING_RATE = 0.001  #0.001
EPOCH = 100      #400 best
BATCH_SIZE = 20    #15
# Using 2 hidden layers  dnn网络
input_size = 21
num_classes = 2
class DNN(nn.Module):    #dnn网络
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 23)
        # self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(23, num_classes)
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
    data = pd.read_excel('q.xls')
    # del data['veil-type']
    # del data['stalk-sha']
    # del data['gill-atta']
    # del data['gill-spa']
    # del data['cap-sur']
    # del data['cap-col']
    # del data['bruises']
    # del data['veil-color']
    # del data['gill-size']
    # del data['ssar']
    # del data['ring-num']
    # del data['habitat']
    # del data['popu']
    # del data['ssbr']
    # del data['ring-typ']


    # del data['spc']
    # del  data['']
    # del data['OBJECTID']
    #print (data.shape[1])
    # data['危险性结果'] = data['危险性结果'].apply(chuliy)
    train, test = train_test_split(data, test_size=0.2, random_state=4)
    #print (test)
    train_label=train['target'].values
    train_data=train.drop(['target'],axis=1).values
    test_label = test['target'].values
    test_data = test.drop(['target'],axis=1).values

    # pca = PCA(n_components=6)
    # train_data = pca.fit_transform(train_data)
    # ratio_pca = pca.explained_variance_ratio_
    # print(pca.explained_variance_ratio_)
    # t_a = 0
    # for a in ratio_pca:
    #     t_a += a
    # print(t_a)
    # # 0.9333174096067566
    # test_data = pca.transform(test_data)

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
    # torch.save(model, './new1.pkl')
    # using crossentropy loss on classification problem



    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = datetime.datetime.now()
    for epoch in range(50):
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
    # torch.save(model, 'E:\pycharpro\model\mg_pca1.pkl')
    end = datetime.datetime.now()
    print(end - start)
if __name__ == '__main__':

    main()







