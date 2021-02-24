import numpy as np

y = np.array([0.1, 0.05, 0.6, 0.0, 0.05])
t = np.array([0, 0, 1, 0, 0])

def sum_squared_error(y, t):
    return 0.5 * np.sum(np.square(y-t))

print(sum_squared_error(y, t))

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

print(cross_entropy_error(y, t))

import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=False,
    one_hot_label=True,
)

print(x_train.shape)
print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(batch_mask)
#print(x_batch)
#print(t_batch)

def cross_entropy_error_oh(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_lbl(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size












