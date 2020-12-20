import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from prunner.mg import DNN
import torch.nn as nn
import pandas as pd
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import datetime

model = torch.load("./mg_prunner.pkl")
print(list(model.named_parameters()))
# print(model.weight)


data = pd.read_excel('q.xls')
train, test = train_test_split(data, test_size=0.2, random_state=4)

train_label = train['target'].values

train_data = train.drop(['target'], axis=1).values
test_label = test['target'].values

test_data = test.drop(['target'], axis=1).values
test_data = torch.Tensor(test_data).to("cpu")
test_label = torch.LongTensor(test_label).to("cpu")
train_data = torch.Tensor(train_data).to("cpu")
train_label = torch.LongTensor(train_label).to("cpu")
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


# optim = torch.optim.Adam(model.parameters(), lr=0.0005)
for epoch in range(1):
    # print(list(model.named_parameters()))
    correct_train = 0
    total_train = 0
    for (data, label) in training_loader:
        pred_label = model(data)
        # loss = criterion(pred_label, label)
        # optim.zero_grad()
        # loss.backward()
        # optim.step()
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
              .format(('%.2f' % (100 * float(correct_test) / float(total_test))), correct_test, total_test))
# torch.save(model, './mg_prunner.pkl')