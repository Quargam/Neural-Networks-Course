import numpy as np
import matplotlib.pyplot as plt


# from scipy.constants import point
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score
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
        Y = np.zeros((point, 1), dtype=np.int8)  # labels vector (0 for red, 1 for blue)

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


class LayerNN:
    """
    neural layer for a neural network
    """

    def __init__(self, lot_of_neuron: int, n_input: int, activation_func):
        self.lot_of_neuron = lot_of_neuron
        self.n_input = n_input
        self.activation_func = activation_func
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

class Optimizer:
    """

    """

    def __init__(self):
        pass


class MultilayerNN:
    """
    neural network
    """

    def __init__(self, param_neural_network: dict,  n_input: list, optimizer):
        """
        :param param_neural_network: a dictionary that describes the number of layers, neurons, and activation functions
        example: neural_network = {"layer_1": [2, 'sigmoid', 'deriv_sigmoid'],
        'layer_2': [3, 'sigmoid', 'deriv_sigmoid'], 'layer_3': [1, 'sigmoid', 'deriv_sigmoid']}
        :param  n_input:
        :param optimizer:
        """
        self.param_neural_network = param_neural_network
        self.lot_of_layer = len(param_neural_network)
        self. n_input = n_input
        self.optimizer = optimizer
        self.neural_network: dict
        self._initialize_params(self.param_neural_network, self. n_input)

    def _initialize_params(self, param_neural_network, n_input):
        """

        :param param_neural_network:
        :param n_input: DOTO: исправить
        :return:
        """
        for key, i in param_neural_network.keys(), range(len(param_neural_network)):
            param_key = param_neural_network[key]
            layer = {key: LayerNN(param_key[0], n_input)}
            self.neural_network.update(layer)


def sigmoid(z):
    z = np.float64(z)
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )

# program body
def main():
    # a = np.array([2])
    # b = np.array([3])
    # neural_network = {"layer_1": [2, 'sigmoid'], 'layer_2': [2, 'sigmoid'], 'layer_3': [2, 'sigmoid']}
    # print(len(neural_network))
    # for _ in range(10):
    #     a = np.vstack((a, [2]))
        # b = np.vstack((b, [3]))
    # print(a.T @ b)
    # print(len(b))
    dataset = Dataset(point=1000, dimensionality=3, petals=4, radius=6)
    dataset.show_start_graph()
    X = np.array([1, 2])
    layer = LayerNN(2, 2, sigmoid)
    layer.forward(X)
    print('end')


if __name__ == '__main__':
    main()
