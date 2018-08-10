import numpy
import matplotlib.pyplot as plt
import scipy.optimize as op


def plot_costs(costs, ax1=None):
    if ax1 is None:
        fig,ax1 = plt.subplots(1, 1)
    for i,c in enumerate(costs):
        ax1.plot(i,c, "r.")
        ax1.set_title("Gradient Descent")
        ax1.set_xlabel("Rounds")
        ax1.set_ylabel("Cost")
    

def plot_data_scatterplot(X, y, thetas=[], costs=None):
    """Plots data as a scatterplot, with contour lines for thetas.
    X: (k, 2) data items.
    y: (k, 1) result (+1 or 0) for each data item in X.
    thetas: list of (theta array, label) pairs to plot contours.
    Plots +1 data points as a green x, -1 as red o.
    """
    #fig, ax = plt.subplots()
    fig,(ax1, ax2) = plt.subplots(1, 2, sharey=False)
    fig.set_tight_layout(True)

    pos=[]
    neg=[]

    pos = [(X[k, 1], X[k, 2]) for k in range(X.shape[0]) if y[k] == 1]
    neg = [(X[k, 1], X[k, 2]) for k in range(X.shape[0]) if y[k] == 0]

    ax1.scatter(*zip(*pos), c='darkgreen', marker='x', linewidths=0)
    ax1.scatter(*zip(*neg), c='red', marker='o', linewidths=0)

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
    	
        contours.append(ax1.contour(xsgrid, ysgrid, plane, colors=colors.next(), levels=[0]))

    if thetas:
        ax1.legend([cs.collections[0] for cs in contours],
                   [label for theta, label in thetas])
    

    if costs is not None:
    	plot_costs(costs, ax2)
    fig.savefig('binary.png', dpi=80)
    plt.show()

def plot_contour(X,Y,thetas, feature_mapping=None, costs=None, lamb=None):

	if costs is None:
		fig, ax1 = plt.subplots(1, 1, sharey=False)
	else:
		fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)

	u = numpy.linspace(-1, 1.5, 50)
	v = numpy.linspace(-1, 1.5, 50)
	
	pos = numpy.where(Y == 1)
	neg = numpy.where(Y == 0)

	ax1.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	ax1.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
	colors = iter(('blue', 'purple', 'black', 'red', 'green', 'orange', "#AABBCC", "#AA22CC",
     "#AA44CC", "#AACCCC", "#32BBCC", "#456CCF", "#AA44CC", "#AACCCC",
     "#AAFF4CC", "#FFC4CC", "#322B2C", "#4565F", "#A344CC", "#A321CC"))

	z = numpy.zeros(shape=(len(u), len(v)))
	contours = []
	for theta, _ in thetas:
		for i in range(len(u)):
			for j in range(len(v)):
				z[i, j] = (feature_mapping(numpy.array(u[i]), numpy.array(v[j])).dot(theta))
		contours.append(ax1.contour(u, v, z, levels=[0],colors=colors.next()))
	


	ax1.set_title('lambda = %f' % lamb)
	ax1.set_xlabel('Microchip Test 1')
	ax1.set_ylabel('Microchip Test 2')
	
	if thetas:
		f=[cs.collections[0] for cs in contours]
		g=[label for theta, label in thetas]
		
		ax1.legend(f,g)
		#plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
	
	if costs is not None:
		plot_costs(costs, ax2)

	fig.savefig('logistic_classification.png', dpi=80)
	plt.show()