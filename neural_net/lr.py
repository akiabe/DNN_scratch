import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

print(f"sigmoid(0) = {sigmoid(0)}")
print(f"sigmoid(9.2) = {sigmoid(9.2)}")

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

tmp_dim = 2
tmp_w, tmp_b = initialize_with_zeros(tmp_dim)
print(f"{tmp_w}, {tmp_b}")

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    grads = {
        "dw": dw,
        "db": db,
    }

    return grads, cost

tmp_w, tmp_b, tmp_X, tmp_Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
tmp_grads, tmp_cost = propagate(tmp_w, tmp_b, tmp_X, tmp_Y)
print(tmp_grads)
print(tmp_cost)
