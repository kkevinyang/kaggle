# coding: utf-8

import pandas as pd # pandas是python的数据格式处理类库

from abupy import AbuML

# 泰坦尼克号生存预测
titanic = AbuML.create_test_more_fiter()
AbuML().estimator.polynomial_regression()