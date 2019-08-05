import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
my_data = pd.read_csv("drug200.csv", delimiter=",")
statinfo = os.stat("drug200.csv")

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

