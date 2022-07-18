# pip install numpy
# pip install opencv-python
# pip install -U scikit-learn
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def read_files(path, ans):
    files = os.listdir(path)
    X = None
    for i, name in enumerate(files):
        img = cv2.imread(path + '/' + name, 0)  # 0 means black-white picture
        if img.shape != 0:
            img = cv2.resize(img, (256, 256))
            vect = img.reshape(1, 256 ** 2) / 255.

            X = vect if (X is None) else np.vstack((X, vect))
        print(f"{i}/{len(files)}")
    print()
    y = np.ones((len(X), 1)) * ans
    return X, y


class LogisticRegression:
    def __init__(self, n_features):
        self.w = np.zeros((1, n_features))
        self.b = np.array([0.])

        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

        self.params = [self.w, self.b]
        self.grads = [self.dw, self.db]

    @staticmethod
    def sigmoid(z):
        return 1. / (np.exp(-z) + 1.)

    def forward(self, X):
        """Method for forward pass implementation

        Input:
            X - matrix of input data, shape (n, n_features)

        Output:
            computed logistic regression of X
        """
        self.X = X
        self.z = X @ self.w.T + self.b
        self.A = self.sigmoid(self.z)

        return self.A

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dL):
        """Method for computing gradient w.r.t. input part of loss function gradient

            Input:
                dL - gradient of loss function w.r.t. the output of the neural network.

            Output:
                None, gradients are accumulated to specific class attribute.
        """
        X = self.X
        assert dL.shape == (1, X.shape[0])

        dw = dL @ (self.A * (1. - self.A) * X)

        db = dL @ (self.A * (1. - self.A) * 1)

        self.dw = self.dw + dw
        self.db = self.db + db


class Optimizer:
    def __init__(self, regression_class, lr=1e-2):
        self.model = regression_class
        self.lr = lr

    def step(self):
        self.model.w = self.model.w - self.lr * self.model.dw
        self.model.b = self.model.b - self.lr * self.model.db

    def zero_grad(self):
        self.model.dw = np.zeros_like(self.model.dw)
        self.model.db = np.zeros_like(self.model.db)


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


def binary_crossentropy(a, y):
    # print(len(a))
    return -(y * np.log(1e-5 + a) + (1. - y) * np.log(1e-5 + 1. - a)).sum() / len(a)


def binary_crossentropy_bw(a, y):
    return -((y - a) / (1e-5 + a * (1. - a))).T / len(a)


def main():
    X_box, y_box = read_files(path='./lesson1_dataset/box/', ans=1.)
    X_no_box, y_no_box = read_files(path='./lesson1_dataset/no_box/', ans=0.)

    X = np.vstack([X_box, X_no_box])
    y = np.vstack([y_box, y_no_box])
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    model = LogisticRegression(n_features=256 ** 2)
    optimizer = Optimizer(model, lr=1e-4)
    loss = Loss(model, binary_crossentropy, binary_crossentropy_bw)

    num_epochs = 500
    for i in range(num_epochs):
        a = model(X_train)
        l = loss(a, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            a_val = model(X_test)
            print("Epoch %d/%d\t Loss: %.3f" % (i, num_epochs, l), end='\t')
            print("Accuracy: %.3f" % (accuracy_score(y_train, a > 0.5)), end='\t')
            print("Val_loss: %.3f" % (loss(a_val, y_test)), end='\t')
            print("Val_accuracy: %.3f" % (accuracy_score(y_test, a_val > 0.5)))


if __name__ == '__main__':
    main()
