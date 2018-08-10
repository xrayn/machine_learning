import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from plotting_lib import *

from logistic_regression_lib import *


def case_1():
    # data_tmp = np.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
    X, Y = load_data('./testdata/ex2data1.txt')
    one = np.ones((len(X), 1))
    # X_normalized=normalize(X)
    X_normalized = X

    # this simply adds 1's in front of the X
    X_mapped = np.concatenate((one, X_normalized), axis=1)
    # X_mapped=feature_mapping(X[:,0],X[:,1])
    theta = np.zeros(((np.shape(X_mapped)[1])))
    # theta=[0.0, 0.0, 0.0]
    gradient_theta, thetas, costs = gradient_descent(
        X_mapped, Y, theta, logistic_loss,
        rounds=10000, alpha=0.001, granularity=10)
    optimal_theta, res = logistic_descent_optimal(X_mapped, Y, theta, lamb=0)
    print "Calculated theta:", gradient_theta
    print "Optimal theta   :", optimal_theta
    print "Theta difference:", gradient_theta - optimal_theta

    thetas.append((optimal_theta, "optimal"))
    plot_data_scatterplot(X_mapped, Y, thetas, costs)


def case_2():
    # data_tmp = np.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
    X, Y = load_data('./testdata/ex2data2.txt')

    X_mapped = feature_mapping(X[:, 0], X[:, 1])
    lamb = 0
    theta = np.zeros(((np.shape(X_mapped)[1])))
    gradient_theta, thetas, costs = gradient_descent(
        X_mapped, Y, theta, logistic_loss,
        rounds=1000, alpha=1, granularity=3, lamb=lamb)
    optimal_theta, res = logistic_descent_optimal(X_mapped, Y, theta, lamb=0)
    print "Calculated theta:", gradient_theta
    print "Optimal theta   :", optimal_theta
    print "Theta difference:", gradient_theta - optimal_theta

    thetas.append((optimal_theta, "optimal"))
    plot_contour(X, Y, thetas, feature_mapping=feature_mapping,
                 costs=costs, lamb=lamb)


def case_3():
    # data_tmp = np.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
    X, Y = load_data('./testdata/ex2data2.txt')

    X_mapped = feature_mapping(X[:, 0], X[:, 1])
    lamb = 0
    theta = np.zeros(((np.shape(X_mapped)[1])))
    thetas = []

    optimal_theta, res = logistic_descent_optimal(X_mapped, Y, theta, lamb=0)
    thetas.append((optimal_theta, "optimal_l0"))
    optimal_theta, res = logistic_descent_optimal(X_mapped, Y, theta, lamb=1)
    thetas.append((optimal_theta, "optimal_l1"))
    print optimal_theta
    # print thetas
    plot_contour(X, Y, thetas, feature_mapping=feature_mapping, lamb=lamb)


case_1()
case_2()
case_3()
