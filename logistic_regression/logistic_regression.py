import numpy
import matplotlib.pyplot as plt

#data_tmp = numpy.genfromtxt('/home/ar/Downloads/machine-learning-ex1/ex1/ex1datatest.txt', delimiter=',')
data = numpy.genfromtxt('./testdata/ex2data1.txt', delimiter=',')
one = numpy.ones((len(data),1))



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
	X=numpy.divide(X,numpy.max(X))
	X=X-numpy.mean(X)
	return X

X_normalized=normalize(X)
#X_normalized=X

X_augment=numpy.concatenate((one, X_normalized), axis=1)


def get_cost(X,Y, theta=[0,0]):
	m=numpy.shape(data)[0]
	h_x=(numpy.dot(X,theta))
	# for logistic regression (classification) the logistic function comes in here!
	h_x_sum=numpy.sum(numpy.subtract(h_x,Y)**2)
	cost=(float(1)/(2*m))*h_x_sum
	return cost



def get_cost_logistic(X,Y, theta=[0,0]):
	m,n=numpy.shape(data)
	h_t_x=(numpy.dot(X,theta))
	eps = numpy.finfo(numpy.float32).eps
	h_x = sigmoid(h_t_x)
	h_x = numpy.clip(h_x, a_min=eps, a_max=1.0-eps)
	
	# for logistic regression (classification) the logistic function comes in here!
	
	h_x_sum=numpy.dot(-Y,numpy.log(h_x))-numpy.dot((1.0-Y),numpy.log(1.0-h_x))
	
	#h_x_2=numpy.dot((1.0-Y),numpy.log(1.0-h_x))
	#h_x_sum=numpy.sum(numpy.subtract(h_x_1,h_x_2))
	cost=(float(1.0)/(m))*h_x_sum
	return cost

def gradient_descent(X,Y,theta,alpha):
	cost_old= get_cost(X,Y,theta)
	h_x=(numpy.dot(X,theta))
	m=numpy.shape(data)[0]
	# h(x)-y
	h_x_substracted=numpy.subtract(h_x,Y)
	theta= theta-(alpha*(1.0/float(m))*numpy.dot(h_x_substracted,X))
	#theta= [theta1, theta2]
	cost= get_cost(X,Y,theta)
	return (theta, cost, cost_old)

def gradient_descent_logistic(X,Y,theta,alpha):
	cost_old= get_cost_logistic(X,Y,theta)
	m=numpy.shape(data)[0]
	
	h_t_x=(numpy.dot(X,theta))
	h_x = sigmoid(h_t_x)
	eps = numpy.finfo(numpy.float32).eps
	h_x = numpy.clip(h_x, a_min=eps, a_max=1.0-eps)
	# h(x)-y
	
	h_x_sum=numpy.subtract(h_x,Y)
	
	#h_x_sum=numpy.dot(h_x_substracted,X)
	
	#theta= [theta1, theta2]
	h_x_substracted=numpy.subtract(h_x,Y)
	theta= theta-(alpha*(1.0/float(m))*numpy.dot(h_x_substracted,X))
	cost= get_cost_logistic(X,Y,theta)
	
	return (theta, cost, cost_old)
def graph(formula, x_range):  
    x = numpy.array(x_range)  
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)  
    


theta=numpy.zeros(((numpy.shape(data)[1]-1)))
theta=[0.0, 0.0, 0.0]

def get_gradient(X,Y,theta, cost_func, descenc_func):
	thetas=[]
	#print get_cost(X,Y, [0,0])
	for i in range(1,1000):
		cost=cost_func(X,Y,theta)
		print cost
		#print get_cost_logistic(X,Y,theta)
		theta, cost, cost_old = descenc_func(X,Y,theta, 1.0)
		if i % 100==1:
			thetas.append((theta, "round"+str(i)))
			plt.plot(i,cost, "r.")
			print i, theta, cost
	thetas.append((theta, "round"+str(i)))
	plt.show()
	return thetas
	

thetas=get_gradient(X_augment,Y,theta,get_cost_logistic, gradient_descent_logistic)



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

    ax.scatter(*zip(*pos), c='darkgreen', marker='x')
    ax.scatter(*zip(*neg), c='red', marker='o', linewidths=0)

    colors = iter(('blue', 'purple', 'black', 'red', 'green', 'orange', "#AABBCC", "#AA22CC", "#AA44CC", "#AACCCC", "#32BBCC", "#456CCF", "#AA44CC", "#AACCCC"))
    
    contours = []
    for theta, _ in thetas:
        xs = numpy.linspace(-1, 2, 200)
        ys = numpy.linspace(-1, 2, 200)
        xsgrid, ysgrid = numpy.meshgrid(xs, ys)
        plane = numpy.zeros_like(xsgrid)
        for i in range(xsgrid.shape[0]):
            for j in range(xsgrid.shape[1]):
                plane[i, j] = numpy.array([1, xsgrid[i, j], ysgrid[i, j]]).dot(
                    theta)
    	
        contours.append(ax.contour(xsgrid, ysgrid, plane, colors=colors.next(),
                                    levels=[0]))

    if thetas:
        plt.legend([cs.collections[0] for cs in contours],
                   [label for theta, label in thetas])
    fig.savefig('binary.png', dpi=80)
    plt.show()

plot_data_scatterplot(X_augment,Y,thetas)

#theta=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#cost= get_cost(X,Y,theta)
#print cost
#print get_cost([[1, 2], [1,3], [1, 4], [1, 5]], [7,6,5,4], [0.1,0.2] )


