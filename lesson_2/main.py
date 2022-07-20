""" python -m pip install -U matplotlib """
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import os


def load_planar_dataset(m=400):
    # np.random.seed(1)
    N = int(m / 2)  # number of points per class
    print(f"N:{N}, m:{m}")
    D = 2  # dimensionality
    X = np.zeros((m, D), dtype=np.float64)  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype=np.int8)  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.14, (j + 1) * 3.12, N, dtype=np.float64) + np.random.randn(N) * 0.4  # theta
        r = 4 * np.sin(a * t) + np.random.randn(N) * 0.1  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    X = X.T
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
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


class OneLayerNN:
    def __init__(self, n_input, n_hidden, activation_funs):
        self.params = {
            "w1": None,
            "w2": None,
            "b1": None,
            "b2": None
        }

        self.gradients = {
            "dw1": None,
            "dw2": None,
            "db1": None,
            "db2": None,
            "dz1": None,
            "dz2": None
        }

        self.cache = {
            "z1": None,
            "a1": None,
            "z2": None,
            "a2": None,
            "X": None
        }

        self._initialize_params(n_input, n_hidden)
        self._initialize_activations(activation_funs)

    def _initialize_params(self, n_input, n_hidden):

        self.params['w1'] = np.random.randn(n_hidden, n_input)
        self.params['w2'] = np.random.randn(1, n_hidden)
        self.params['b1'] = np.zeros((n_hidden,))
        self.params['b2'] = np.zeros((1,))

        for key in self.params.keys():
            self.gradients[key] = np.zeros_like(self.params[key])

    def _initialize_activations(self, activation_funs):
        for i, activation in enumerate(activation_funs):
            if activation == "tanh":
                self.__dict__['activation' + str(i + 1)] = np.tanh
                self.__dict__['activation' + str(i + 1) + "_backward"] = \
                        lambda x: 1 - np.tanh(x) ** 2
            if activation == "sigmoid":
                self.__dict__['activation' + str(i + 1)] = self.sigmoid
                self.__dict__['activation' + str(i + 1) + "_backward"] = \
                    lambda x: self.sigmoid(x) * (1. - self.sigmoid(x))

    @staticmethod
    def sigmoid(z):
        z = np.float64(z)
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

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

        z1 = X @ self.params['w1'].T + self.params['b1']
        # print(f"X:{X.shape},self.params['w1'].T:{self.params['w1'].T.shape}, z1:{z1.shape}")
        a1 = self.activation1(z1)
        # print(f"a1:{a1.shape}")

        z2 = a1 @ self.params['w2'].T + self.params['b2']
        a2 = self.activation2(z2)
        # print(f"a2:{a2.shape},self.params['w2'].T:{self.params['w2'].T.shape}, z2:{z2.shape}")
        self.collect_cache(z1=z1, z2=z2, a1=a1, a2=a2, X=X)

        return a2

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dL):

        z1, a1, z2, a2, X = self.unpack_cache(['z1', 'a1', 'z2',
                                               'a2', 'X'])

        dz2 = dL * (a2 * (1. - a2))  # dL * div_sigmoid(a2)
        dw2 = dL.T @ a1
        db2 = dL.T @ np.ones((z2.shape[0]))

        dz1 = (dz2 @ self.params['w2']) * self.activation2_backward(z1)
        dw1 = (dz1.T @ X)
        db1 = dz1.T @ np.ones(X.shape[0])

        self.collect_grads(dw2=dw2, dw1=dw1, db2=db2, db1=db1)

    def summary(self):
        num_layers = len(self.params) // 2
        for i in range(1, num_layers + 1):
            w_ind, b_ind = f'w{i}', f'b{i}'
            print(f"Layer_{i}\t weights shape: {self.params[w_ind].shape}\t bias shape: {self.params[b_ind].shape}")


class Optimizer:
    def __init__(self, regression_class, lr=1e-2):
        self.model = regression_class
        self.lr = lr

    def step(self):
        new_params = {
            k: None for k in self.model.params
        }
        params = self.model.params
        grads = self.model.gradients
        # print("PARAMS:",params)
        # print("GRADS:",grads)
        for key in self.model.params.keys():
            new_params[key] = params[key] - self.lr * grads["d" + key]
        self.model.params = new_params

    def zero_grad(self):
        for key in self.model.gradients.keys():
            self.model.gradients[key] = np.zeros_like(self.model.gradients[key])


class Loss:
    def __init__(self, model, loss_fn, loss_fn_bw):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_fn_bw = loss_fn_bw
        self.dL = None
        self.a = None
        self.y = None

    def __call__(self, a, y):
        self.a = a
        self.y = y
        return self.loss_fn(a, y)

    def forward(self, a, y):
        return self.__call__(a, y)

    def backward(self, **params):
        assert (self.a is not None) and (self.y is not None), "loss.forward() must be called first!"
        self.dL = self.loss_fn_bw(self.a, self.y)
        self.model.backward(self.dL, **params)


def binary_crossentropy(a, y, eps=1e-5):
    # print(len(a))
    # return -(y * np.log(eps + a) + (1. - y) * np.log(eps + 1. - a)).mean()
    return ((y - a) ** 2).mean()


def binary_crossentropy_bw(a, y, eps=1e-5):
    return -1. / len(a) * ((y - a) / (eps + a * (1. - a)))


def main_example():
    X, Y = load_planar_dataset()  # загружаем датасет
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=40, cmap=plt.cm.Spectral)
    plt.show()

    from sklearn.linear_model import LogisticRegression  # IMPORT MODEL FROM SKLEARN

    clf = LogisticRegression(solver='lbfgs')  # INITIALIZE MODEL
    clf.fit(X, Y[:, 0])  # FIT THE MODEL

    plot_decision_boundary(lambda x: clf.predict(x), X, Y[:, 0])
    # plt.title(u"Логистическая регрессия")
    plt.show()
    # Print accuracy
    LR_predictions = clf.predict(X)
    print(u'Качество модели (согласно метрике accuracy): %.2f ' % (accuracy_score(Y[:, 0], LR_predictions)))


# class Linearization:
#     def __init__(self, params):
#         self.params = {}
#         for key in params.keys():
#             self.params[key] = params[key]
#
#     def forward(self, X):
#         z1 = X @ self.params['w1'].T + self.params['b1']
#         a1 = z1
#
#         z2 = a1 @ self.params['w2'].T + self.params['b2']
#         a2 = z2
#         return a2
#
#     def __call__(self, X):
#         return self.forward(X)


def main():
    X, y = load_planar_dataset(m=1000)

    print(f"X shape is {X.shape},\ty shape is {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    model = OneLayerNN(n_input=2, n_hidden=32, activation_funs=['tanh', 'sigmoid'])  # ['tanh', 'sigmoid']
    optimizer = Optimizer(model, lr=1e-2)
    loss = Loss(model, binary_crossentropy, binary_crossentropy_bw)
    model.summary()

    num_epochs = 15000
    for i in range(num_epochs):
        a = model(X_train)
        l = loss(a, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i % 400 == 0):
            a_val = model(X_test)
            print("Epoch %d/%d\t Loss: %.3f" % (i, num_epochs, l), end='\t')
            print("Accuracy: %.3f" % (accuracy_score(y_train, a > 0.5)), end='\t')
            print("Val_loss: %.3f" % (loss(a_val, y_test)), end='\t')
            print("Val_accuracy: %.3f" % (accuracy_score(y_test, a_val > 0.5)))

    print("Decision Boundary for train, hidden layer size " + str(model.params['w1'].shape[0]))
    plot_decision_boundary(lambda x: model(x) > 0.5, X_train, y_train)

    print("Decision Boundary for test, hidden layer size " + str(model.params['w1'].shape[0]))
    plot_decision_boundary(lambda x: model(x) > 0.5, X_test, y_test)

    model.params


if __name__ == '__main__':
    # main_example()
    main()
