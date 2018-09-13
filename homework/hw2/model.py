import numpy as np
import time
from math import sqrt
from random import randint
from scipy.signal import convolve2d
from activate_functions import relu, gradient_for_relu, softmax

class ConvolutionalNeuralNetwork:
    def __init__(self, input_dim, num_classes=10, filter_size=3, stride=1, padding=0, num_channels=1):
        """Simple CNN with 1 filter (#activation maps=1)"""
        self.d = input_dim
        self.k = num_classes
        self.filter_size = filter_size
        self.s = stride
        self.p = padding
        self.c = num_channels

        self.out_dim = self.d - self.filter_size + 1

        self.filter = np.random.randn(self.filter_size, self.filter_size) * sqrt(2.0 / (input_dim * input_dim))
        self.w = np.random.rand(self.k, self.out_dim, self.out_dim) * sqrt(2.0 / (input_dim * input_dim))
        self.b = np.zeros(self.k)
    
    def train(self, train_data, learning_rate=0.1, epochs=100):
        X = train_data[0]
        Y = train_data[1]

        # avg_epochs = epochs // 10  # Should tune this in practice.
        epoch_limit = 7

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Learning rate schedule.
            if epoch <= epoch_limit:
                learning_rate = learning_rate
            elif epoch > epoch_limit:
                learning_rate = 0.00001
            elif epoch > 2 * epoch_limit:
                learning_rate = 0.000001
            else:
                learning_rate = 0.0000001
           
            # SGD.
            total_correct = 0
            for _ in range(X.shape[0]):
                index = randint(0, X.shape[0] - 1)
                x = X[index]
                y = Y[index]
                
                # Backpropagation.
                z, h, u, f = self._forward_step(x)
                g_b, g_w, g_f = self._backward_step(x, y, z, h, u, f)
                self._update_weights(learning_rate, g_b, g_w, g_f)

                # Current training accuracy.
                if self._predict(f) == y:
                    total_correct += 1
            
            acc = total_correct / np.float(X.shape[0])
            print("epoch {}, training accuracy = {}".format(epoch, acc))

            # Record time.
            end_time = time.time()
            print("--- %s seconds ---" % (end_time - start_time))
    
    def _forward_step(self, x):
        # Reshape x to a matrix.
        x = self._reshape_x_to_matrix(x)

        # Forward step.
        z = convolve2d(x, self.filter, mode="valid")
        h = relu(z)
        u = np.sum(np.multiply(self.w, h), axis=(1,2)) + self.b
        f = softmax(u)

        return z, h, u, f
    
    def _reshape_x_to_matrix(self, x):
        return np.reshape(x, (self.d, self.d))

    def _backward_step(self, x, y, z, h, u, f):
        x = self._reshape_x_to_matrix(x)

        e_y = np.zeros(self.k)
        e_y[y] = 1
        gradient_u = - (e_y - f)
        gradient_b = gradient_u

        gradient_w = np.zeros((self.k, self.out_dim, self.out_dim))
        for i in range(self.k):
            gradient_w[i] = gradient_u[i] * h
        
        grad_u_times_w = np.multiply(np.reshape(gradient_u, (self.k, 1, 1)), self.w)
        delta = grad_u_times_w.sum(axis=0)
        relu_prime = gradient_for_relu(z)
        gradient_filter = convolve2d(x, np.multiply(relu_prime, delta), mode="valid")

        # print(gradient_filter.shape)

        return gradient_b, gradient_w, gradient_filter
    
    def _update_weights(self, learning_rate, g_b, g_w, g_f):
        self.b -= learning_rate * g_b
        self.w -= learning_rate * g_w
        self.filter -= learning_rate * g_f
    
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