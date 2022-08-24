import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import point
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# import os


class Dataset:
    """
    pass
    """

    def __init__(self, point=400, radius=4, petals=4, random_seed=None, dimensionality=2, random_dist_points=6):
        self.point = point  # number of points
        self.radius = radius  # radius ray of the flower
        self.petals = petals  # maximum ray of the flower
        self.random_seed = random_seed  # generating points
        self.dimensionality = dimensionality  # dimensionality
        self.random_dist_points = random_dist_points  # randomness of the distribution of points
        self.X, self.Y = self.load_planar_dataset(self.point, self.radius, self.petals,
                                                  self.random_seed, self.dimensionality, self.random_dist_points)

    def load_planar_dataset(self, point, radius, petals, random_seed, dimensionality, random_dist_points):
        np.random.seed(random_seed)
        N = int(point / dimensionality)  # number of points per class
        D = dimensionality  # dimensionality
        X = np.zeros((point, 2), dtype=np.float64)  # data matrix where each row is a single example
        Y = np.zeros((point, 1), dtype=np.float64)  # labels vector (0 for red, 1 for blue)

        for j in range(D):
            ix = range(N * j, N * (j + 1))
            t = np.linspace(j * (6.28 / D), (j + 1) * (6.28 / D), N,
                            dtype=np.float64) + np.random.randn(N) * (D / random_dist_points)  # theta
            r = radius * np.sin(petals * t) + np.random.randn(N) * 0.2
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j

        return X, Y  # X - coordinates of points; Y - color of points

    def show_start_graph(self):
        plt.figure(figsize=(10, 6))
        plt.grid()  # сетка на графике
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y[:, 0], s=40, cmap=plt.cm.Spectral)
        plt.show()


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    X = X.T
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.03
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


class LayerNN:
    """
    neural layer for a neural network
    """

    def __init__(self, n_input: int, lot_of_neuron: int, activation_func=lambda x: x,
                 deriv_activation_func=lambda x: 1):
        """
        :param n_input: number of input features for this layer
        :param lot_of_neuron: number of layer neurons
        :param activation_func: activation function
        :param deriv_activation_func: gradient activation function
        """
        self.lot_of_neuron = lot_of_neuron
        self.n_input = n_input
        self.activation_func = activation_func  # activation function
        self.deriv_activation_func = deriv_activation_func  # derivative of the activation function
        self.params = {
            "w1": None,
            "b1": None,
        }
        self.gradients = {
            "dw1": None,
            "db1": None,
            "dz1": None,
        }
        self.cache = {
            "z1": None,
            "a1": None,
            "X": None
        }
        self._initialize_params(n_input, lot_of_neuron)

    def _initialize_params(self, n_input, n_hidden):

        self.params['w1'] = np.random.randn(n_hidden, n_input)
        self.params['b1'] = np.zeros((n_hidden,))

        for key in self.params.keys():
            self.gradients[key] = np.zeros_like(self.params[key])

    def collect_cache(self, **kwargs):
        for key in kwargs.keys():
            self.cache[key] = kwargs[key]

    def collect_grads(self, **kwargs):
        for key in kwargs.keys():
            self.gradients[key] = kwargs[key]

    def unpack_cache(self, keys):
        output = []
        for key in keys:
            output.append(self.cache[key])
        output = tuple(output)
        return output

    def forward(self, X):
        """Method for forward pass implementation

        Input:
            X - matrix of input data, shape (n, n_features)

        Output:
            computed logistic regression of X
        """
        w1 = self.params['w1']
        b1 = self.params['b1']
        z1 = X @ w1.T + b1
        a1 = self.activation_func(z1)
        self.collect_cache(z1=z1, a1=a1, X=X)
        return a1

    def backward(self, dL):
        """
        calculates the gradient for the layer
        :param dL:
        :return:
        """
        z1, a1, X = self.unpack_cache(['z1', 'a1', 'X'])
        dz1 = dL * self.deriv_activation_func(a1)
        dw1 = dz1.T @ X
        db1 = dz1.T @ np.ones((z1.shape[0]))
        self.collect_grads(dw1=dw1, db1=db1)
        return dz1 @ self.params["w1"]


class Optimizer:
    """
    changes weights and b
    """

    def __init__(self, model, lr=1e-2):
        """
        :param model: neural network object
        :param lr: assessment of training
        """
        self.model = model
        self.lr = lr

    def step(self):
        """
        updates weights and b by gradient descent
        :return: None
        """
        for key in self.model.neural_network.keys():
            old_params = self.model.neural_network[key].params
            new_params = {
                k: None for k in old_params
            }
            params = old_params
            grads = self.model.neural_network[key].gradients
            for i in old_params.keys():
                new_params[i] = params[i] - self.lr * grads["d" + i]
                # print(f'{key}:\t{i}:{params[i].shape}')
            self.model.neural_network[key].params = new_params

    def zero_grad(self):
        """
        zeroes the gradient values
        :return: None
        """
        for _, key in enumerate(self.model.neural_network.keys(), 0):
            for i in self.model.neural_network[key].gradients.keys():
                self.model.neural_network[key].gradients[i] = np.zeros_like(self.model.neural_network[key].gradients[i])


class MultilayerNN:
    """
    neural network
    """

    def __init__(self, param_neural_network: dict, n_input: int, ):
        """
        :param param_neural_network: a dictionary that describes the number of layers, neurons, activation functions and
                                                                               gradient activation functions
        example: neural_network = {'layer_1': [10, 'sigmoid', 'deriv_sigmoid'],
                                   'layer_2': [30, 'sigmoid', 'deriv_sigmoid'],
                                   'last_layer_3': [1, 'sigmoid', 'deriv_sigmoid']}
        :param  n_input: number of input parameters(number of features)
        """
        self.param_neural_network = param_neural_network  # neural network parameters
        self.n_input = n_input  # number of features
        self.neural_network = {}  # stored layers of the neural network
        self.outputNN = None  # the vector of values that the neural network outputs
        self._initialize_layers(self.param_neural_network, self.n_input)

    def _initialize_layers(self, param_neural_network, n_input):
        """
        creates layers of a neural network
        :return: outputs a dictionary with keys layer names and values with objects of the LayerNN class
        """
        n_input_arr = []
        n_input_arr.append(n_input)
        for i, key in enumerate(param_neural_network.keys(), 0):
            param_key = param_neural_network[key]
            n_input_arr.append(param_key[0])
            layer = {key: LayerNN(n_input_arr[i], param_key[0], param_key[1], param_key[2])}
            self.neural_network.update(layer)
        return self.neural_network

    def forward(self, X):
        """
        calculates the output of a neural network
        :param X: input data vector
        :return: the vector of values that the neural network outputs
        """
        a = X
        for i, key in enumerate(self.param_neural_network.keys(), 0):
            a = self.neural_network[key].forward(a)
        self.outputNN = a
        return self.outputNN

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dL):
        """
        passes the gradient of the function from one layer to another for back propagation
        :param dL: the value of the gradient of the loss function
        :return: None
        """
        dA = dL
        for i, key in enumerate(reversed(self.neural_network.keys()), 0):
            dA = self.neural_network[key].backward(dA)


class Loss:
    """
    Loss class that counts the loss function, forward and backward of model neural_network
    """

    def __init__(self, model, loss_fn, loss_fn_bw):
        """
        :param model: model neural_network
        :param loss_fn: loss function
        :param loss_fn_bw:derivative of the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.loss_fn_bw = loss_fn_bw
        self.dL = None
        self.a = None
        self.y = None

    def __call__(self, a, y):
        """
        :param a: vector of predictions
        :param y: vector of observed
        :return: the value of the MSE function
        """
        self.a = a
        self.y = y
        return self.loss_fn(a, y)

    def forward(self, a, y):
        return self.__call__(a, y)

    def backward(self, **params):
        """
        passes dL to back propagation
        :return: vector of values of the gradient of the loss function
        """
        assert (self.a is not None) and (self.y is not None), "loss.forward() must be called first!"
        self.dL = self.loss_fn_bw(self.a, self.y)
        self.model.backward(self.dL, **params)
        return self.dL


def sigmoid(z):
    """
    sigmoid function
    :param z: vector of values
    :return: the value of the sigmoid function
    """
    z = np.float64(z)
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def deriv_sigmoid(z):
    """
    derivative of the sigmoid function
    :param z: vector of values
    :return: the gradient value of the sigmoid
    """
    z = np.float64(z)
    return sigmoid(z) * (1. - sigmoid(z))


def mse_func(a, y):
    """
    MSE(mean squared error) function
    :param a: vector of predictions
    :param y: vector of observed
    :return: the value of the MSE function
    """
    return ((y - a) ** 2).mean()


def deriv_mse_func(a, y):
    """
    derivative of the MSE(mean squared error) function
    :param a: vector of predictions
    :param y: vector of observed
    :return: the gradient value of the MSE function
    """
    return -1. / len(a) * (y - a) * 2


def binary_crossentropy(a, y, eps=1e-5):
    return -(y * np.log(eps + a) + (1. - y) * np.log(eps + 1. - a)).mean()


def binary_crossentropy_bw(a, y, eps=1e-5):
    return -1. / len(a) * ((y - a) / (eps + a * (1. - a)))


"""program body"""


def main():
    # neural_network = {"layer_1": [20, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
    #                   'last_layer_2': [1, sigmoid, deriv_sigmoid]}
    # neural_network = {"layer_1": [20, sigmoid, deriv_sigmoid],
    #                   'last_layer_2': [1, sigmoid, deriv_sigmoid]}
    neural_network = {"layer_1": [200, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
                      'layer_2': [150, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
                      'layer_3': [100, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
                      'last_layer_6': [1, sigmoid, deriv_sigmoid]}
    # neural_network = {"layer_1": [15, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
    #                   'layer_2': [35, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
    #                   'last_layer_3': [1, sigmoid, deriv_sigmoid]}
    # neural_network = {"layer_1": [15, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
    #                   'layer_2': [30, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
    #                   'layer_3': [45, np.tanh, lambda x: 1 - np.tanh(x) ** 2],
    #                   'last_layer_4': [1, sigmoid, deriv_sigmoid]}
    dataset = Dataset(point=600, dimensionality=2, petals=4, radius=6, random_seed=None)
    X, y = dataset.X, dataset.Y
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    # dataset.show_start_graph()
    model = MultilayerNN(neural_network, n_input=2, )
    optimizer = Optimizer(model, lr=1e-2)
    # loss = Loss(model, binary_crossentropy, binary_crossentropy_bw)
    loss = Loss(model, mse_func, deriv_mse_func)
    x_list_plt = []
    y_list_train_plt = []
    y_list_test_plt = []
    num_epochs = 5000
    for i in range(num_epochs):
        a = model.forward(X_train)
        l = loss(a, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if ((i % 100 == 0)):
            a_test = model.forward(X_test)
            l_test = loss(a_test, y_test)
            x_list_plt.append(i)
            y_list_train_plt.append(l)
            y_list_test_plt.append(l_test)
            a_val = model(X_test)
            print("Epoch %d/%d\t Loss: %.3f" % (i, num_epochs, l), end='\t')
            print("Accuracy: %.3f" % (accuracy_score(y_train, a > 0.5)), end='\t')
            print("Val_loss: %.3f" % (loss(a_val, y_test)), end='\t')
            print("Val_accuracy: %.3f" % (accuracy_score(y_test, a_val > 0.5)))

    'сильно лагает когда грузит график'
    plt.title('train & test')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    y_list_train_plt_min = min(y_list_train_plt)
    x_list_train_plt_min = x_list_plt[y_list_train_plt.index(y_list_train_plt_min)]
    plt.annotate('min', xy=(x_list_train_plt_min, y_list_train_plt_min), xycoords='data',
                 xytext=(x_list_train_plt_min, y_list_train_plt_min * 1.3), textcoords='data',
                 arrowprops=dict(facecolor='b'))
    plt.plot(x_list_plt, y_list_train_plt, label='train')
    plt.plot(x_list_plt, y_list_test_plt, label='test')
    plt.legend()
    plt.show()
    plot_decision_boundary(lambda x: model(x) > 0.5, X_train, y_train)
    # plot_decision_boundary(lambda x: model(x), X_train, y_train)
    # print("Decision Boundary for test, hidden layer size ")
    # plot_decision_boundary(lambda x: model(x) > 0.5, X_test, y_test)


if __name__ == '__main__':
    main()
