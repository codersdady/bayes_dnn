from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from graphviz import Digraph
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = BayesianModel([('feel', 'buying'),
                       ('feel', 'persons'),
                       ('feel', 'maint'),
                       ('feel', 'safety'),
                       ('feel', 'doors'),
                       ('feel', 'lug_boot'),
                       ('buying', 'maint'),
                       ('buying', 'safety'),
                       ('safety', 'lug_boot'),
                       ('lug_boot', 'doors')])
train=pd.read_csv('./car.data')

test=pd.read_csv('./test.txt')

model.fit(train, estimator=BayesianEstimator, prior_type="K2")

def showBN(model, save=False):
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

predict_data=test.drop(columns=['feel'],axis='1')
re=pd.read_csv('./re.txt')
# print(re.info())
# print(predict_data.info())

y_pred = model.predict(predict_data)
# 预测结果

# print(model.get_cpds()) 各个节点条件概率情况
# re['doors'] = re['doors'].astype('object')

# print(model.predict_probability(re))
# 预测概率

# print((y_pred['feel']==test['feel']).sum()/len(test))  准确率