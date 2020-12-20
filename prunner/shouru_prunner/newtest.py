import time
from sklearn.decomposition import PCA
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from prunner.mg import DNN
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import datetime
import scipy.sparse as sparse
import csv

# model = torch.load("E:\pycharpro\model\mg_pca1.pkl")
model = torch.load("E:\pycharpro\model\sr0.pkl")
# print(model)
# print(list(model.named_parameters()))

# print(list(model.named_parameters()))
# num=sparse.coo_matrix(moudel.weight.detach().numpy())
# print(moudel.weight)
# print(num.toarray())
# print(type(moudel.weight))
# torch.save(model, './mg_prunner.pkl')
# prune.remove(moudel, 'weight')



# prune.remove(moudel, 'weight')


# print(list(moudel.named_parameters()))


# print(model)

train = pd.read_excel('./data.xlsx')
test = pd.read_excel('./test.xlsx')


# train, test = train_test_split(data, test_size=0.2, random_state=4)
#
train_label = train['target'].values
train_data = train.drop(['target'], axis=1).values
test_label = test['target'].values
test_data = test.drop(['target'], axis=1).values



# print(test_data)

# pca = PCA(n_components=5)
# train_data = pca.fit_transform(train_data)
#
# ratio_pca = pca.explained_variance_ratio_
# print(pca.explained_variance_ratio_)
# t_a = 0
# for a in ratio_pca:
#     t_a += a
# print(t_a)
# 0.9333174096067566
# test_data = pca.transform(test_data)

# np.save('E:\\pycharpro\\prunner\\jinxuan_prunner\\train_data',train_data)
# np.save('E:\\pycharpro\\prunner\\jinxuan_prunner\\test_data',test_data)
# np.save('E:\\pycharpro\\prunner\\jinxuan_prunner\\test_label',test_label)
# np.save('E:\\pycharpro\\prunner\\jinxuan_prunner\\train_label',train_label)

# print(type(test_data))
# train_label = np.load('E:\\pycharpro\\prunner\\shouru_prunner\\train_label.npy')
# train_data = np.load('E:\\pycharpro\\prunner\\shouru_prunner\\train_data.npy')
# test_label = np.load('E:\\pycharpro\\prunner\\shouru_prunner\\test_label.npy')
# test_data = np.load('E:\\pycharpro\\prunner\\shouru_prunner\\test_data.npy')
# print(type(test_data))
# print(train_data)
test_data = torch.Tensor(test_data).to("cpu")
test_label = torch.LongTensor(test_label).to("cpu")
train_data = torch.Tensor(train_data).to("cpu")
train_label = torch.LongTensor(train_label).to("cpu")
# we want to use GPU if we have one

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prepare the data loader
training_set = Data.TensorDataset(train_data,
                                  train_label)
training_loader = Data.DataLoader(dataset=training_set,
                                  batch_size=70,
                                  shuffle=True)
testing_set = Data.TensorDataset(test_data,
                                 test_label)
testing_loader = Data.DataLoader(dataset=testing_set,
                                 batch_size=70,
                                 shuffle=False)


# using crossentropy loss on classification problem
criterion = nn.CrossEntropyLoss()


optim = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(100):

for epoch in range(500):
    # print(list(model.named_parameters()))
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
    moudel = model.fc1
    prune.l1_unstructured(moudel, name="weight", amount=0.86)
    prune.remove(moudel, 'weight')
    moudel = model.fc3
    prune.l1_unstructured(moudel, name="weight", amount=0.86)
    prune.remove(moudel, 'weight')
    # print('Epoch {:3d} Accuracy on training data: {}% ({}/{})'
    #       .format(epoch, (100 * correct_train / total_train), correct_train, total_train))
    # pytorch 0.4 feature, not calculate grad on test set
    # 预测阶段，不跟新权值
    # start = time.time()
    # with torch.no_grad():
    #     correct_test = 0
    #     total_test = 0
    #     for (data, label) in testing_loader:
    #         pred_label = model(data)
    #         _, answer = torch.max(pred_label.data, 1)
    #         total_test += label.size(0)
    #         correct_test += (answer == label).sum()
    #     print('          Accuracy on testing data: {}% ({}/{})'
    #           .format(('%.2f' % (100 * float(correct_test) / float(total_test))), correct_test, total_test))
# torch.save(model, './mg_prunner.pkl')
# print(list(model.named_parameters()))
# print(list(model.named_parameters()))
# print(testing_loader)
start = time.time()
with torch.no_grad():
    correct_test = 0
    total_test = 0
    for (data, label) in testing_loader:
        pred_label = model(data)
        # break
        _, answer = torch.max(pred_label.data, 1)
        total_test += label.size(0)
        correct_test += (answer == label).sum()
end = time.time()
print(end - start)
    # print('          Accuracy on testing data: {}% ({}/{})'
    #       .format(('%.2f' % (100 * float(correct_test) / float(total_test))), correct_test, total_test))

# end = time.process_time()
# print(end - start)
print('          Accuracy on testing data: {}% ({}/{})'
      .format(('%.2f' % (100 * float(correct_test) / float(total_test))), correct_test, total_test))
# print(end - start)
# print(list(model.named_parameters()))