import numpy as np
import random
import matplotlib.pyplot as plt
# import activation function
from activationf import sigmoid, softmax, relu, tanh, derivative_relu, derivative_tanh

X_train = np.loadtxt('cats_data/cat_train_x.csv', delimiter=',') / 255.0
Y_train = np.loadtxt('cats_data/cat_train_y.csv', delimiter=',').reshape(1, X_train.shape[1])
X_test = np.loadtxt('cats_data/cat_test_x.csv', delimiter=',') / 255.0
Y_test = np.loadtxt('cats_data/cat_test_y.csv', delimiter=',').reshape(1, X_test.shape[1])


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
            layer_dims[l - 1])  # *0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def forward_prop(X, parameters, activation):
    forward_cache = {}
    L = len(parameters) // 2
    forward_cache['A0'] = X

    for l in range(1, L):
        forward_cache["Z" + str(l)] = parameters["W" + str(l)].dot(forward_cache["A" + str(l-1)]) + parameters["b" + str(l)]
        if activation == "tanh":
            forward_cache["A" + str(l)] = tanh(forward_cache["Z" + str(l)])
        else:
            forward_cache["A" + str(l)] = relu(forward_cache["Z" + str(l)])

    forward_cache['Z' + str(L)] = parameters['W' + str(L)].dot(forward_cache['A' + str(L - 1)]) + parameters[
        'b' + str(L)]

    if forward_cache['Z' + str(L)].shape[0] == 1:
        forward_cache['A' + str(L)] = sigmoid(forward_cache['Z' + str(L)])
    else:
        forward_cache['A' + str(L)] = softmax(forward_cache['Z' + str(L)])

    return forward_cache['A' + str(L)], forward_cache


def compute_cost(AL, Y):
    m = Y.shape[1]
    if Y.shape[0] == 1:
        cost = -(1./m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1 - AL).T))
    else:
        cost = -(1./m) * np.sum(Y * np.log(AL))
    cost = np.squeeze(cost)
    return cost

def back_prop(AL, Y, parameters, forward_cache, activation):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[1]

    grads["dZ" + str(L)] = AL - Y
    grads["dW" + str(L)] = (1/m) * np.dot(grads["dZ" + str(L)], forward_cache["A" + str(L-1)].T)
    grads["db" + str(L)] = (1/m) * np.sum(grads["dZ" + str(L)], axis=1, keepdims=True)

    if activation == "tanh":
        activation_d = derivative_tanh
    else:
        activation_d = derivative_relu
    for l in reversed(range(1, L)):
        grads["dZ" + str(l)] = np.dot(parameters["W" + str(l+1)].T, grads["dZ" + str(l+1)]) * activation_d(forward_cache['A' + str(l)])
        grads["dW" + str(l)] = (1/m) * np.dot(grads["dZ" + str(l)], forward_cache["A" + str(l-1)].T)
        grads["db" + str(l)] = (1/m) * np.sum(grads["dZ" + str(l)], axis=1, keepdims=True)

    return grads




def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, y, parameters, activation):
    m = X.shape[1]
    y_pred, caches = forward_prop(X, parameters, activation)

    if y.shape[0] == 1:
        y_pred = np.array(y_pred > 0.5, dtype='float')
    else:
        y = np.argmax(y, 0)
        y_pred = np.argmax(y_pred, 0)

    return np.round(np.sum((y_pred == y) / m), 2)


def model(X, Y, layers_dims, learning_rate=0.03, activation='relu', num_iterations=3000):  # lr was 0.009
    costs = []
    parameters = initialize_parameters(layers_dims)
    aL, forw_cache = forward_prop(X_train, parameters, 'relu')

    for l in range(len(parameters) // 2 + 1):
        print("Shape of A" + str(l) + " :", forw_cache['A' + str(l)].shape)

    for i in range(0, num_iterations):
        AL, forward_cache = forward_prop(X, parameters, activation)
        cost = compute_cost(AL, Y)
        gradients = back_prop(AL, Y, parameters, forward_cache, activation)
        parameters = update_parameters(parameters, gradients, learning_rate)
        if i % (num_iterations / 10) == 0:
            print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:{}".format(
                i, np.round(cost, 2),
                predict(X_train, Y_train, parameters, activation),
                predict(X_test, Y_test, parameters, activation)
            ))
        if i % 10 == 0:
            print("==", end="")
    print()
    return parameters

def predict_n_random_image(X, y, parameters, activation, times):
    y_pred, caches = forward_prop(X, parameters, activation)
    y_pred = np.squeeze(y_pred)
    for _ in range(times):
        index = random.randrange(0, X.shape[1])
        image = X[:, index].reshape(64, 64, 3)
        plt.imshow(X[:, index].reshape(64, 64, 3))
        plt.show()
        m = X.shape[1]
        if y.shape[0] == 1:
            print(y_pred[index] > 0.5)
        else:
            y = np.argmax(y, 0)
            y_pred = np.argmax(y_pred, 0)
            print(y_pred[index])


if __name__ == "__main__":
    # row pixels (rgb) col  images
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    layers_dims = [X_train.shape[0], 64, 64, 16,  Y_train.shape[0]]
    lr = 0.0015
    iters = 4000
    activation = "relu"
    parameters = model(X_train, Y_train, layers_dims, learning_rate=lr, activation=activation, num_iterations=iters)
    predict_n_random_image(X_test, Y_test,parameters,activation, 10)
