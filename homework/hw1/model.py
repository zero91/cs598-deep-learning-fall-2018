import numpy as np
from random import randint
from activate_functions import relu, gradient_for_relu, softmax

class NeuralNetwork:
    def __init__(self, dimension, hidden_units=100, classes=10):
        """Init a fully-connected neural network with a single hidden layer.
        Args:
            dimension(int): dimension of an input sample x
            hidden_units(int): number of hidden units
            classes(int): number of classes in y
        """
        self.d = dimension
        self.d_h = hidden_units
        self.k = classes

        self.w = np.random.randn(self.d_h, self.d)
        self.b1 = np.random.randn(self.d_h)
        self.c = np.random.randn(self.k, self.d_h)
        self.b2 = np.random.randn(self.k)
    
    def train(self, train_data, test_data, learning_rate=0.1, epochs=100):
        """Train the neural network using SGD.
        Args:
            train_data(tuple): x_train(60000, 784) and y_train(60000,)
            learning_rate(float)
            epochs(int)
        """
        X = train_data[0]
        Y = train_data[1]

        # avg_epochs = epochs // 10  # Should tune this in practice.
        avg_epochs = epochs // 10

        for epoch in range(1, epochs + 1):
            # Learning rate schedule.
            if epoch > avg_epochs:
                learning_rate = 0.01
            elif epoch > 2 * avg_epochs:
                learning_rate = 0.001
            else:
                learning_rate = 0.0001
           
            # SGD.
            total_correct = 0
            for _ in range(X.shape[0]):
                index = randint(0, X.shape[0] - 1)
                x = X[index]
                y = Y[index]
                
                # Backpropagation.
                z, h, u, f = self._forward_step(x)
                g_c, g_b2, g_b1, g_w = self._backward_step(x, y, z, h, u, f)
                self._update_weights(learning_rate, g_c, g_b2, g_b1, g_w)

                # Current training accuracy.
                if self._predict(f) == y:
                    total_correct += 1
            
            acc = total_correct / np.float(X.shape[0])
            print("epoch {}, training accuracy = {}".format(epoch, acc))

            if epoch % 5 == 0:
                self.test(test_data)

    
    def _forward_step(self, x):
        """Calculate output f and intermediary network values."""
        z = np.matmul(self.w, x) + self.b1
        h = relu(z)
        u = np.matmul(self.c, h) + self.b2
        f = softmax(u)
        return z, h, u, f

    def _backward_step(self, x, y, z, h, u, f):
        """Calculate the gradient w.r.t parameters."""
        e_y = np.zeros(self.k)
        e_y[y] = 1
        gradient_u = - (e_y - f)

        gradient_b2 = gradient_u
        gradient_c = np.matmul(gradient_u[:, np.newaxis], h[np.newaxis, :])  # col * row

        delta = np.matmul(self.c.T, gradient_u)
        relu_prime = gradient_for_relu(z)
        gradient_b1 = np.multiply(delta, relu_prime)
        gradient_w = np.matmul(gradient_b1[:, np.newaxis], x[np.newaxis, :])  # col * row

        return gradient_c, gradient_b2, gradient_b1, gradient_w
    
    def _update_weights(self, learning_rate, g_c, g_b2, g_b1, g_w):
        self.c -= learning_rate * g_c
        self.b2 -= learning_rate * g_b2
        self.b1 -= learning_rate * g_b1
        self.w -= learning_rate * g_w
    
    def _predict(self, f):
        return np.argmax(f)
   
    def test(self, test_data):
        X = test_data[0]
        Y = test_data[1]
        total_correct = 0

        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            _, _, _, f = self._forward_step(x)
            if self._predict(f) == y:
                total_correct += 1
        
        acc = total_correct / np.float(X.shape[0])
        print("testing accuracy = {}".format(acc))
        