import numpy
import matplotlib.pyplot as plt
import scipy.optimize as op


#data_tmp = numpy.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
data = numpy.genfromtxt('./testdata/ex2data2.txt', delimiter=',')
one = numpy.ones((len(data),1))

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

X,Y=split_x_y(data)

def normalize(X):
	mean=X.mean(axis=0)
	std=X.std(axis=0)
	#X=numpy.divide(X,numpy.max(X))
	X=(X-mean)/std
	
	return X


def normalize2(X):
	X=numpy.divide(X,numpy.max(X))
	#X=X-numpy.mean(X)
	
	x= numpy.linalg.svd(X)
	
	return X

#X_normalized=normalize(X)
X_normalized=X


# this simply adds 1's in front of the X
X_augment=numpy.concatenate((one, X_normalized), axis=1)
X_mapped=feature_mapping(X[:,0],X[:,1])


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

def logistic_cost(X,Y,theta):
	"""
	Calculates the cost of the current values
	"""
	m,n=X.shape
	h_x=logistic_loss(X,theta)
	h_x_sum=numpy.dot(-Y,numpy.log(h_x))-numpy.dot((1.0-Y),numpy.log(1.0-h_x))
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
	gradient= numpy.dot(h_x_substracted,X)/m
	
	return gradient


def logistic_gradient_descent_round(X,Y,theta,alpha):
	"""
	Calculates new values for theta.
	Applies learning rate alpha to the calculated gradient and
	subtracts it from the current theta.

	Returns: (theta) where theta is one step closer to the optima
	"""	
	cost_before=logistic_cost(X,Y,theta)
	theta= theta-(alpha*logistic_gradient(X,Y,theta))
	cost=logistic_cost(X,Y,theta)
	return theta, cost, cost_before, (cost_before-cost)


def logistic_cost_wrap(theta, X,Y):
	return logistic_cost(X,Y,theta)

def logistic_gradient_wrap(theta, X,Y):
	return logistic_gradient(X,Y,theta)


def graph(formula, x_range):  
    x = numpy.array(x_range)  
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)  
    


theta=numpy.zeros(((numpy.shape(X_mapped)[1])))
#theta=[0.0, 0.0, 0.0]

def gradient_descent(X,Y,theta, lossfunc, rounds=1000):
	"""
	This is a self implemented gradient descent function.
	"""
	thetas=[]
	costs=[]
	#theta,cost,costdelta = logistic_gradient_descent_round(X,Y,theta,  0.0)
	#thetas.append((theta, "start"))
	for i in range(1,rounds):
		theta,cost,cost_before, costdelta = logistic_gradient_descent_round(X,Y,theta,  1)
		costs.append(cost)
		if i % (rounds/3)==1:
			thetas.append((theta, "round"+str(i)))
			#plt.plot(i,costs[i-1], "r.")
			print i, theta, cost, cost_before, costdelta
			plt.plot(i,cost, "r.")
	#thetas.append((theta, "round"+str(i)))
	plt.show()
	return theta,thetas
	

gradient_theta, thetas=gradient_descent(X_mapped ,Y,theta, logistic_loss, 100)



def logistic_descent_optimal(X, Y, theta):
	"""
	This uses a more sophisticated descent algorithm (TNC) from scipy.
	"""
	Result = op.minimize(fun=logistic_cost_wrap, x0 = theta, args = (X, Y), method = 'TNC', jac = logistic_gradient_wrap);
	optimal_theta = Result.x;
	
	return Result.x,Result

optimal_theta, res =logistic_descent_optimal(X_mapped,Y, theta)
print "Calculated theta:", gradient_theta
print "Optimal theta   :",optimal_theta
print "Theta difference:",gradient_theta - optimal_theta


thetas.append((optimal_theta,"optimal"))
#graph(lambda x: x*theta[2]+x*theta[1]+x*theta[0], range(-1, 3))	
#plt.plot(X[:,1],Y, "ro")

#plt.show()	

def plot_data_scatterplot(X, y, thetas=[]):
    """Plots data as a scatterplot, with contour lines for thetas.
    X: (k, 2) data items.
    y: (k, 1) result (+1 or 0) for each data item in X.
    thetas: list of (theta array, label) pairs to plot contours.
    Plots +1 data points as a green x, -1 as red o.
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    pos=[]
    neg=[]

    pos = [(X[k, 1], X[k, 2]) for k in range(X.shape[0]) if y[k] == 1]
    neg = [(X[k, 1], X[k, 2]) for k in range(X.shape[0]) if y[k] == 0]

    ax.scatter(*zip(*pos), c='darkgreen', marker='x', linewidths=0)
    ax.scatter(*zip(*neg), c='red', marker='o', linewidths=0)

    colors = iter(('blue', 'purple', 'black', 'red', 'green', 'orange', "#AABBCC", "#AA22CC",
     "#AA44CC", "#AACCCC", "#32BBCC", "#456CCF", "#AA44CC", "#AACCCC",
     "#AAFF4CC", "#FFC4CC", "#322B2C", "#4565F", "#A344CC", "#A321CC"))
    
    contours = []
    for theta, _ in thetas:
    	xmax=round(X.max()+1)
    	xmin= round(X.min()-1)
        xs = numpy.linspace(xmin, xmax, 200)
        ys = numpy.linspace(xmin, xmax, 200)
        xsgrid, ysgrid = numpy.meshgrid(xs, ys)
        plane = numpy.zeros_like(xsgrid)
        for i in range(xsgrid.shape[0]):
            for j in range(xsgrid.shape[1]):
                plane[i, j] = numpy.array([1, xsgrid[i, j], ysgrid[i, j]]).dot(
                    theta)
    	
        contours.append(ax.contour(xsgrid, ysgrid, plane, colors=colors.next(), levels=[0]))

    if thetas:
        plt.legend([cs.collections[0] for cs in contours],
                   [label for theta, label in thetas])
    fig.savefig('binary.png', dpi=80)
    plt.show()

def plot_contour(X,Y,theta):
	u = numpy.linspace(-1, 1.5, 50)
	v = numpy.linspace(-1, 1.5, 50)
	
	pos = numpy.where(Y == 1)
	neg = numpy.where(Y == 0)

	plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
	colors = iter(('blue', 'purple', 'black', 'red', 'green', 'orange', "#AABBCC", "#AA22CC",
     "#AA44CC", "#AACCCC", "#32BBCC", "#456CCF", "#AA44CC", "#AACCCC",
     "#AAFF4CC", "#FFC4CC", "#322B2C", "#4565F", "#A344CC", "#A321CC"))

	z = numpy.zeros(shape=(len(u), len(v)))
	contours = []
	for theta, _ in thetas:
		for i in range(len(u)):
			for j in range(len(v)):
				z[i, j] = (feature_mapping(numpy.array(u[i]), numpy.array(v[j])).dot(theta))
		contours.append(plt.contour(u, v, z, levels=[0],colors=colors.next()))
	


	plt.title('lambda = %f' % 1)
	plt.xlabel('Microchip Test 1')
	plt.ylabel('Microchip Test 2')
	
	if thetas:
		f=[cs.collections[0] for cs in contours]
		g=[label for theta, label in thetas]
		
		plt.legend(f,g)
		#plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
	plt.show()

plot_contour(X,Y, thetas)
#plot_data_scatterplot(X_augment,Y,thetas)
#print X_augment
#print logistic_loss((numpy.array([1,45,85])), optimal_theta)
#plot_data_scatterplot(X_augment,Y,[thetas[len(thetas)-1]])

#theta=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#cost= get_cost(X,Y,theta)
#print cost
#print get_cost([[1, 2], [1,3], [1, 4], [1, 5]], [7,6,5,4], [0.1,0.2] )


