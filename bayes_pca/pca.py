import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

train = pd.read_excel('./data.xlsx')
test = pd.read_excel('./test.xlsx')

train_label = train['target'].values
test_label = test['target'].values
train=train.drop(['target'],axis=1)
test=test.drop(['target'],axis=1)

pca = PCA(n_components=10)
train = pca.fit_transform(train)
ratio_pca = pca.explained_variance_ratio_
# print(pca.explained_variance_ratio_)
t_a = 0
for a in ratio_pca:
    t_a += a
# print(t_a)
test=pca.transform(test)


data = pd.DataFrame(test)
data=data.reset_index(drop=True)
ct = pd.DataFrame({'target': test_label})
final=pd.concat([data,ct],axis=1)
final.to_csv("pca_test.csv", index=False,encoding = "GB2312")

data = pd.DataFrame(train)
data=data.reset_index(drop=True)
ct = pd.DataFrame({'target': train_label})
final=pd.concat([data,ct],axis=1)
final.to_csv("pca_train.csv", index=False,encoding = "GB2312")




