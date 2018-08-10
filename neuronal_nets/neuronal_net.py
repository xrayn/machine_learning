import scipy.io
import numpy as np
import sys


sys.path.append('../logistic_regression')
from logistic_regression_lib import *

mat = scipy.io.loadmat('data/ex3data1.mat')

X = mat["X"]
y = mat["y"]

print(mat["X"].shape)
print(X[0])
print(mat["y"].shape)
print(y[0])
matheta = scipy.io.loadmat('data/ex3weights.mat')

theta1 = matheta["Theta1"]
theta2 = matheta["Theta2"]


class NeuronalNet:
    def __init__(self, x, w1, w2, y):
        self.input_layer = x
        self.output_layer = []
        self.hidden_layer = None
        self.y = y
        self.w1 = w1
        self.w2 = w2
        self.m, self.n = self.input_layer.shape
        self.cost = 0

        # the network
        #
        #  input    w1        hidden
        #   0 ----- w1_0 --- (hidden) --- w2_0 --- (output)
        #   0 ----- w1_1 --- (hidden) --- w2_1 --- (output)
        #   0 ----- w1_2 --- (hidden) --- w2_2 --- (output)

        # self.hidden_layer=sigmoid(self.input_layer.T.dot(w1.T))
        # self.hidden_layer=self.add_1s_hidden_layers(self.hidden_layer)
        # self.output_layer=sigmoid(self.hidden_layer.T.dot(w1.T))
    def derivate(self, x):
        return x * (1 - x)

    def logistic_cost(self, h_x, Y):
        """
        Calculates the cost of the current values
        """
        m, n = self.input_layer.shape

        h_x_sum = np.dot(-Y, np.log(h_x)) - \
            np.dot((1.0 - Y), np.log(1.0 - h_x))

        # regularized theta does not consider the thetha[0]
        # thetaR = theta[1:]
        # lambda_term=((lamb)/2.0*m)*thetaR.dot(thetaR)
        # h_x_sum=h_x_sum+lambda_term
        cost = (float(1.0) / (m)) * h_x_sum
        self.cost += cost
        return cost

    def get_output(self):
        return self.output_layer

    def logistic_cost_regularized(self, lamb=1):

        theta_1_sum = 0
        theta_2_sum = 0
        # this executes the sum for 25*400
        for i in range(self.w1.shape[0]):
            for j in range(self.w1.shape[1]):
                thetaR1 = self.w1[i][j]
                theta_1_sum += thetaR1 * thetaR1

        for i in range(self.w2.shape[0]):
            for j in range(self.w2.shape[1]):
                thetaR2 = theta2[i][j]
                theta_2_sum += thetaR2 * thetaR2

        # this executes the sum for 10*25

        lambda_term = ((lamb) / (2.0 * self.m)) * (theta_1_sum + theta_2_sum)
        print((lamb) / (2.0 * self.m))
        print((theta_1_sum + theta_2_sum))
        print(lambda_term)
        self.cost_regularized = lambda_term
        return lambda_term

    def full_feed_forward(self):
        res=[]
        for i in range(self.input_layer.shape[0]):
            forward_vector = self.input_layer[i]
            index, value = nn.feed_forward(forward_vector)
            y_t = np.zeros(10)
            # y_t[self.y-1]=1.0
            y_t[self.y[i] - 1] = 1.
        # print(y_t)
            m, n = self.input_layer.shape
            self.logistic_cost(self.output_layer, y_t)
            res.append((index, value))
        return res

    def feed_forward(self, forward_vector):
        self.hidden_layer = sigmoid(forward_vector.dot(self.w1.T))
        self.hidden_layer = add_1s_hidden_layers(self.hidden_layer)
        self.output_layer = sigmoid(self.hidden_layer.T.dot(self.w2.T))

        # first transform y to a vector
        res = (np.argmax(self.output_layer), np.max(self.output_layer))

        return res

    def get_cost(self):
        return self.cost

    def get_cost_regularized(self):
        return self.cost + self.cost_regularized

    def add_1s(self, X):
        m, n = X.shape
        one = np.ones((m, 1))
        # this simply adds 1's in front of the X
        X = np.concatenate((one, X), axis=1)
        return X

    def add_1s_hidden_layers(layer):
        return np.insert(layer, 0, 1., axis=0)


def add_1s(X):
    m, n = X.shape
    one = np.ones((m, 1))
    # this simply adds 1's in front of the X
    X = np.concatenate((one, X), axis=1)
    return X


def add_1s_hidden_layers(layer):
    return np.insert(layer, 0, 1., axis=0)


# this simply adds 1's in front of the X
X = add_1s(X)
print()

epsilon_init = 0.12


print(theta1.shape)
w1_neurons = 25
w1 = np.random.rand(w1_neurons, X.shape[1]) * 2 * epsilon_init - epsilon_init
print(w1.shape)
w2_neurons = 10
print(theta2.shape)
w2 = np.random.rand(w2_neurons, w1.shape[0] + 1) * 2 * epsilon_init - epsilon_init

print("w2", w2)


def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))



nn = NeuronalNet(X, theta1, theta2, y)
#nn = NeuronalNet(X, w1, w2, y)
cost = 0.0
nn.full_feed_forward()

def backprop():
    # Step 1
    forward_vector = X[0]
    nn.feed_forward(forward_vector)

    y_t = np.zeros(10)
    y_t[y[0] - 1] = 1.
    # output_delta=(nn.output_layer - y_t)
    # print output_delta

    # # Step 2
    # # z^2 =(theta.dot)
    # hidden_delta=w2.T.dot(output_delta)*sigmoid_gradient(nn.hidden_layer)
    # hidden_delta=hidden_delta[1:]
    # print "hidden_delta", hidden_delta

    # we now have d^2 and d^3

    # calculate the gradient
    # 
    a_1 = forward_vector
    z_2 = a_1.dot(w1.T)
    a_2 = sigmoid(z_2)
    z_3 = a_2.dot(w2.T[1:])
    a_3 = sigmoid(z_3)


    d_3=a_3-y_t
    print "d3", d_3.shape, "w2", w2.shape, "z2", z_2.shape
    d_2=w2.T*d_3
    print d_2.shape
    print d_2.T.shape

    d_2=d_2.T*sigmoid_gradient(add_1s_hidden_layers(z_2))


    print a_2.shape



# for i in range(X.shape[0]):
#     index, value = nn.feed_forward(i)

#     if not y[i] - 1 == index:
#         print(y[i] - 1, index, value, "ERROR")
#     else:
#         print(y[i] - 1, index, value, "OK")

nn.logistic_cost_regularized(1)

cost = nn.get_cost()
print("Cost [" + str(cost) + "]")

cost_regularized = nn.get_cost_regularized()
print("Cost regularized[" + str(cost_regularized) + "]")


# ok=0
# err=0
#
# for j in range(X.shape[0]):
#   idx, value = predict(X[j,:],theta1,theta2)
#
#   if (idx+1) == y[j]:
#       ok=ok+1.0
#   else:
#       err=err+1.0
# print( (ok/5000*100), (err/5000*100))

exit()
