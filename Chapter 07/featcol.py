import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.contrib import learn

N = 10000

weight = np.random.randn(N)*5+70
spec_id = np.random.randint(0,3,N)
bias = [0.9,1,1.1]
height = np.array([weight[i]/100 + bias[b] for i,b in enumerate(spec_id)])
spec_name = ['Goblin', 'Human', 'Manbears']
spec = [spec_name[s] for s in spec_id]

df = pd.DataFrame({'Species':spec, 'Weight':weight, 'Height':height})

from tensorflow.contrib import layers
Weight = layers.real_valued_column("Weight")

Species = layers.sparse_column_with_keys(column_name="Species", keys=['Goblin', 'Human', 'Manbears'])

# reg = learn.LinearRegressor(feature_columns=[Weight, Species])
reg = tf.estimator.LinearRegressor(feature_columns=[Weight, Species])


def input_fn(df):
    feature_cols = {}
    feature_cols['Weight'] = tf.constant(df['Weight'].values)

    feature_cols['Species'] = tf.SparseTensor(indices=[[i, 0] for i in range(df['Species'].size)], values=df['Species'].values, dense_shape=[df['Species'].size, 1])

    labels = tf.constant(df['Height'].values)

    return feature_cols, labels

    # SparseTensor(indices=[[0, 0], [2, 1], [2, 2]], values=[2, 5, 7], dense_shape=[3, 3])
    # [0, 0] -> in the first row, put the first element
    # [2, 1] -> in the third row, put the second element
    # [2, 2] -> in the third row, put the third element

# reg.fit(input_fn=lambda:input_fn(df), steps=500)
reg.train(input_fn=lambda:input_fn(df), steps=50000)
# print(reg.get_variable_names())

# w_w = reg.get_variable_value('linear/Weight/weight')
w_w = reg.get_variable_value('linear/linear_model/Weight/weights')
print('Estimation for Weight: {}'.format(w_w))

# s_w = reg.get_variable_value('linear/Species/weights')
s_w = reg.get_variable_value('linear/linear_model/Species/weights')
# b = reg.get_variable_value('linear/bias_weight')
b = reg.get_variable_value('linear/linear_model/bias_weights')
print('Estimation for Species: {}'.format(s_w + b))
