import random
import numpy as np
import matplotlib.pyplot as plt


def display_random_digit(X_train):
    index = int(random.randrange(0, X_train.shape[1]))
    plt.imshow(X_train[:, index].reshape((28, 28)), cmap='gray')
    plt.show()


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    expX = np.exp(x)
    return expX/np.sum(expX, axis=0)


def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def derivative_relu(x):
    return np.array(x > 0, dtype = np.float32)


# nr of neurons in x layer, in hidden layer and in y layer
def initialize_parameters(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros((n_y, 1))
    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters


def forward_prop(x, parameters):
    # To fetch the parameters
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
   # print("w1 shape ", w1.shape)
    z1 = np.dot(w1, x) + b1
  #  print("z1 shape ", z1.shape)
    a1 = tanh(z1)
   # print("a1 shape ", a1.shape)

    z2 = np.dot(w2, a1) + b2
  #  print("z2 shape ", z2.shape)
    a2 = softmax(z2)

    forward_cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2
    }

    return forward_cache


def backward_prop(x, y, parameters, forward_cache):
    m = x.shape[1]

    # Fetching our parameters
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # Fetching our forward_cache
    a1 = forward_cache['a1']
    a2 = forward_cache['a2']

    dz2 = a2 - y
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * derivative_tanh(a1)
    dw1 = (1/m) * np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2
    }

    return gradients


def cost_function(a2, y):
    m = y.shape[1]
    return - (1/m * np.sum(y * np.log(a2)))


def update_parameters(parameters, gradients, learning_rate):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = gradients['dw1']
    db1 = gradients['db1']
    dw2 = gradients['dw2']
    db2 = gradients['db2']

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters


def model(x, y, n_h, learning_rate, iterations):
    n_x = x.shape[0]
    n_y = y.shape[0]
    list_costs = []
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range (iterations):
        forward_cache = forward_prop(x, parameters)
        cost = cost_function(forward_cache['a2'], y)
        gradients = backward_prop(x, y, parameters, forward_cache)
        parameters = update_parameters(parameters, gradients, learning_rate)
        list_costs.append(cost)

        if i % (iterations / 10) == 0:
            print("Cost after", i, "iterations is :", cost)

    return parameters, list_costs


def accuracy(inp, labels, parameters):
    forward_cache = forward_prop(inp, parameters)
    a_out = forward_cache['a2']  # containes propabilities with shape(10, 1)

    a_out = np.argmax(a_out, 0)  # 0 represents row wise

    labels = np.argmax(labels, 0)

    acc = np.mean(a_out == labels) * 100

    return acc


if __name__ == "__main__":
    X_train = np.loadtxt('train_X.csv', delimiter=',')
    Y_train = np.loadtxt('train_label.csv', delimiter=',')
    X_train = X_train.T
    Y_train = Y_train.T

    X_test = np.loadtxt('test_X.csv', delimiter=',')
    Y_test = np.loadtxt('test_label.csv', delimiter=',')
    X_test = X_test.T
    Y_test = Y_test.T
    # shape n, m    rows = each pixel  columns = each image
    print("shape of X_train : ", X_train.shape)
    print("shape of Y_train : ", Y_train.shape)

    print("shape of X_test : ", X_test.shape)
    print("shape of Y_test : ", Y_test.shape)
    iterations = 100
    n_h = 1000
    learning_rate = 0.02
    Parameters, Cost_list = model(X_train, Y_train, n_h=n_h, learning_rate=learning_rate, iterations=iterations)
    t = np.arange(0, iterations)
    plt.plot(t, Cost_list)
    plt.show()

    print("Accuracy of Train Dataset", accuracy(X_train, Y_train, Parameters), "%")
    print("Accuracy of Test Dataset", round(accuracy(X_test, Y_test, Parameters), 2), "%")
    for _ in range(10):
        idx = int(random.randrange(0, X_test.shape[1]))
        plt.imshow(X_test[:, idx].reshape((28, 28)), cmap='gray')
        plt.show()

        cache = forward_prop(X_test[:, idx].reshape(X_test[:, idx].shape[0], 1), Parameters)
        a_pred = cache['a2']
        a_pred = np.argmax(a_pred, 0)

        print("Our model says it is :", a_pred[0])
