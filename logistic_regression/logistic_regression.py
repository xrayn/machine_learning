import numpy
import matplotlib.pyplot as plt
import scipy.optimize as op

from plotting_lib import *



def feature_mapping(X1,X2):
	#extract X1 and X2
	
	#X1=X[:,0]
	#X2=X[:,1]
	X1.shape = (X1.size, 1)
	X2.shape = (X2.size, 1)
	out = numpy.ones(shape=(X1[:, 0].size, 1))
	degree = 6
	for i in range(1,degree+1):
		for j in range(i+1):
			r = (X1 ** (i - j)) * (X2 ** j)
			out=numpy.append(out,r, axis=1)
			
	return out

def sigmoid(z):
	s=1.0/(1.0+numpy.exp(-z))
	return s


def split_x_y(data):
	columns=numpy.shape(data)[1]
	rows=numpy.shape(data)[0]
	X=data[:, :(columns-1)]
	Y=data[:,(columns-1)]
	
	#adjust
	#print numpy.mean(X)
	#print numpy.max(X)
	#X=numpy.divide(X,numpy.max(X))
	#X=X-numpy.mean(X)
	
	return (X,Y)



def normalize(X):
	mean=X.mean(axis=0)
	std=X.std(axis=0)
	#X=numpy.divide(X,numpy.max(X))
	X=(X-mean)/std
	
	return X


def get_cost_quadratic(X,Y, theta=[0,0]):
	m=numpy.shape(data)[0]
	h_x=(numpy.dot(X,theta))
	
	h_x_sum=numpy.sum(numpy.subtract(h_x,Y)**2)
	cost=(float(1)/(2*m))*h_x_sum
	return cost

def logistic_loss(X,theta):
	"""
	Calculates the h(x) logistic regression.
	In this case function h() is sigmoid

	Returns: (h_x) where sigmoid is applied for h_x
	"""
	h_x=numpy.dot(X,theta)
	# we apply sigmoid function here to get a bowl curve for GD
	h_x=sigmoid(h_x)
	# this calculates the cost for logistic regression
	return h_x

def logistic_cost(X,Y,theta, lamb=1):
	"""
	Calculates the cost of the current values
	"""
	m,n=X.shape
	h_x=logistic_loss(X,theta)
	h_x_sum=numpy.dot(-Y,numpy.log(h_x))-numpy.dot((1.0-Y),numpy.log(1.0-h_x))
	cost=(float(1.0)/(m))*h_x_sum
	return cost

def logistic_cost_regularized(X,Y,theta, lamb=1):
	"""
	Calculates the cost of the current values
	"""
	m,n=X.shape
	h_x=logistic_loss(X,theta)
	h_x_sum=numpy.dot(-Y,numpy.log(h_x))-numpy.dot((1.0-Y),numpy.log(1.0-h_x))
	
	#regularized theta does not consider the thetha[0]	
	thetaR = theta[1:]
	lambda_term=((lamb)/2.0*m)*thetaR.dot(thetaR)
	h_x_sum=h_x_sum+lambda_term
	
	cost=(float(1.0)/(m))*h_x_sum
	
	return cost


def logistic_gradient(X,Y,theta):
	"""
	Calculates the gradient for logistic regression
	"""
	eps = numpy.finfo(numpy.float32).eps
	m,n=X.shape
	h_x = logistic_loss(X,theta)
	# need some clipping of values since it might overflow otherwise
	h_x = numpy.clip(h_x, a_min=eps, a_max=1.0-eps)
	h_x_substracted=numpy.subtract(h_x,Y) # sigmod of x_theta - y

	# this is the general form of gradient decent.
	# thetas are derived on the basis of the used loss function
	
	# this is correct for the non-regularized case
	gradient= numpy.dot(h_x_substracted,X)/float(m)
	
	return gradient

def logistic_gradient_regularized(X,Y,theta, lamb=1):
	"""
	Calculates the gradient for logistic regression
	"""
	m,n=X.shape
	normal_gradient=logistic_gradient(X,Y,theta)
	
	# we want to regularize the normal gradient.
	# so we generate a reduced thetaR theta(1,n)
	thetaR = theta[1:]
	
	#then we take the normal_gradient(1,n)
	#print lamb,m, (float(lamb)/float(m)), normal_gradient[1:]
	#calculate the delta based on lambda (l/m)
	delta=normal_gradient.dot((float(lamb)/float(m)))
	#set the first delta to be 0 (this value should remain as it is in the gradient)
	delta[0]=0

	# and add the delta values to the original gradient
	# ?it is unclear whether we need to devide by m in this case or not.?
	# ?for the contour it does not make much of a difference.?
	
	# it is etiher:
	gradient=(normal_gradient+(delta/float(m)))
	# or:
	#gradient=(normal_gradient+(delta))
	# ^-- a different case though is when the logistic_gradient function does not devide by m,
	#     which is wrong for lambda=0 but may be OK for when we regularize here.

	return gradient


	
def logistic_gradient_descent_round(X,Y,theta,alpha, lamb=1):
	"""
	Calculates new values for theta.
	Applies learning rate alpha to the calculated gradient and
	subtracts it from the current theta.

	Returns: (theta) where theta is one step closer to the optima
	"""	
	m,n=X.shape
	cost_before=logistic_cost(X,Y,theta)
	
	theta= theta-(alpha*logistic_gradient(X,Y,theta))
	cost=logistic_cost(X,Y,theta)
	return theta, cost, cost_before, (cost_before-cost)


def logistic_cost_wrap(theta, X,Y, lamb=1):
	return logistic_cost_regularized(X,Y,theta, lamb)

def logistic_gradient_wrap(theta, X,Y, lamb=1):
	return logistic_gradient_regularized(X,Y,theta,lamb)


def logistic_cost_wrap2(theta, X,Y, lamb=1):
	return logistic_cost(X,Y,theta)

def logistic_gradient_wrap2(theta, X,Y, lamb=1):
	return logistic_gradient(X,Y,theta)


def graph(formula, x_range):  
    x = numpy.array(x_range)  
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)  
    

def gradient_descent(X,Y,theta, lossfunc, rounds=1000, alpha=1, granularity=10, lamb=1):
	"""
	This is a self implemented gradient descent function.
	"""
	thetas=[]
	costs=[]
	#theta,cost,costdelta = logistic_gradient_descent_round(X,Y,theta,  0.0)
	#thetas.append((theta, "start"))
	for i in range(1,rounds):
		theta,cost,cost_before, costdelta = logistic_gradient_descent_round(X,Y,theta,  alpha, lamb)
		if i % (rounds/granularity)==1:
			costs.append(cost)
			thetas.append((theta, "round"+str(i)))
			#plt.plot(i,costs[i-1], "r.")
			print i, theta, cost, cost_before, costdelta
			#plt.plot(i,cost, "r.")
	#thetas.append((theta, "round"+str(i)))
	
	return theta,thetas, costs

def logistic_descent_optimal(X, Y, theta, lamb):
	"""
	This uses a more sophisticated descent algorithm (TNC) from scipy.
	"""
	Result = op.minimize(fun=logistic_cost_wrap, x0 = theta, args = (X, Y, lamb), method = 'TNC', jac = logistic_gradient_wrap);
	optimal_theta = Result.x;
	
	return Result.x,Result

def logistic_descent_optimal2(X, Y, theta):
	"""
	This uses a more sophisticated descent algorithm (TNC) from scipy.
	"""
	Result = op.minimize(fun=logistic_cost_wrap2, x0 = theta, args = (X, Y), method = 'TNC', jac = logistic_gradient_wrap2);
	optimal_theta = Result.x;
	
	return Result.x,Result

def load_data(filename):
	data = numpy.genfromtxt(filename, delimiter=',')
	
	X,Y=split_x_y(data)
	return X,Y

def case_1():
	#data_tmp = numpy.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
	X,Y=load_data('./testdata/ex2data1.txt')
	one = numpy.ones((len(X),1))
	#X_normalized=normalize(X)
	X_normalized=X

	# this simply adds 1's in front of the X
	X_mapped=numpy.concatenate((one, X_normalized), axis=1)
	#X_mapped=feature_mapping(X[:,0],X[:,1])
	theta=numpy.zeros(((numpy.shape(X_mapped)[1])))
	#theta=[0.0, 0.0, 0.0]
	gradient_theta, thetas, costs=gradient_descent(X_mapped ,Y,theta, logistic_loss, rounds=10000, alpha=0.001, granularity=10)
	optimal_theta, res =logistic_descent_optimal(X_mapped,Y, theta)
	print "Calculated theta:", gradient_theta
	print "Optimal theta   :",optimal_theta
	print "Theta difference:",gradient_theta - optimal_theta

	thetas.append((optimal_theta,"optimal"))
	plot_data_scatterplot(X_mapped,Y,thetas, costs)

def case_2():
	#data_tmp = numpy.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
	X,Y=load_data('./testdata/ex2data2.txt')

	X_mapped=feature_mapping(X[:,0],X[:,1])
	lamb=0
	theta=numpy.zeros(((numpy.shape(X_mapped)[1])))
	gradient_theta, thetas, costs=gradient_descent(X_mapped ,Y,theta, logistic_loss, rounds=1000, alpha=1, granularity=3, lamb=lamb)
	optimal_theta, res =logistic_descent_optimal(X_mapped,Y, theta)
	print "Calculated theta:", gradient_theta
	print "Optimal theta   :",optimal_theta
	print "Theta difference:",gradient_theta - optimal_theta

	thetas.append((optimal_theta,"optimal"))
	plot_contour(X,Y, thetas, feature_mapping=feature_mapping, costs=costs, lamb=lamb)



def case_3():
	#data_tmp = numpy.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
	X,Y=load_data('./testdata/ex2data2.txt')

	X_mapped=feature_mapping(X[:,0],X[:,1])
	lamb=0
	theta=numpy.zeros(((numpy.shape(X_mapped)[1])))
	thetas=[]
	optimal_theta, res =logistic_descent_optimal2(X_mapped,Y, theta)
	thetas.append((optimal_theta,"optimal_old"))

	optimal_theta, res =logistic_descent_optimal(X_mapped,Y, theta, lamb=0)
	thetas.append((optimal_theta,"optimal_l0"))
	optimal_theta, res =logistic_descent_optimal(X_mapped,Y, theta, lamb=1)
	thetas.append((optimal_theta,"optimal_l1"))
	print optimal_theta
	#print thetas
	plot_contour(X,Y, thetas, feature_mapping=feature_mapping, lamb=lamb)


case_3()
