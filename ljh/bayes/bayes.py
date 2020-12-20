from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from graphviz import Digraph
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start=time.process_time()
model = BayesianModel([('scene', 'distant'),
                       ('scene', 'light'),
                       ('scene', 'en_light'),
                       ('scene', 're'),
                       ('distant', 'light'),
                       ('light', 'en_light')])
train=pd.read_excel('./6.xlsx')
test=pd.read_excel('./6.xlsx')

model.fit(train, estimator=BayesianEstimator, prior_type="K2")

def showBN(model, save=True):
    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2'
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    edges=model.edges
    for a, b in edges:
        dot.edge(a, b)
    if save:
        dot.view(cleanup=True)
    return dot

predict_data=test.drop(columns=['scene'],axis='1')
# re=pd.read_csv('./re.txt')
# print(re.info())
# print(predict_data.info())
print("预测数据集")
print(predict_data)
y_pred = model.predict(predict_data)
showBN(model)
print("预测结果")
print(y_pred)
# 预测结果

print("节点条件概率情况")
print(model.get_cpds())
# 各个节点条件概率情况
# re['doors'] = re['doors'].astype('object')

# print(model.predict_probability(re))
# 预测概率
print("预测准确率")
print((y_pred['scene']==test['scene']).sum()/len(test))
end=time.process_time()
print("总运行时间：")
print('Running time: %s Seconds'%(end-start))
# 准确率