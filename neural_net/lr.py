import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten=True,
    normalize=False,
)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    return s

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0

    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print(f"cost after iteration : {i}, {cost}")

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.