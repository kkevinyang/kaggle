# coding: utf-8
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt

x = np.arange(-3.0, 6.0, 0.1)
scores = np.vstack

# 函数y = 1*x^2 + 0*x + 0
y = np.poly1d([1, 0, 0])
