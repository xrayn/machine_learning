import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import collections
import pickle
import os.path

from plotting_lib import *

from logistic_regression_lib import *

from mnist import MNIST
#(self, path='.', mode='vanilla', return_type='lists', gz=False)
mndata = MNIST('./../mnist_data', return_type="numpy")

mndata.gz = True



#import pyculib as pycu

def predict(x,theta):
	#return sigmoid(pycu.blas.dot(x,theta))
	return sigmoid(x.dot(theta))

def norm(x, should_normalize=False):
	if should_normalize:
		print "Applying Normalization"
		return normalize(x)
	print "Not Applying Normalization"
	return x

#X=normalize(X)
def train_digits(retrain=False, theta_file="mnist_thetas.pickle", do_normalize=False, lamb=1.0):
	if os.path.isfile(theta_file) and not retrain:
		#print "File {} exists. Loading thetas".format(theta_file)
		thetas = pickle.load( open( theta_file, "rb" ) )
	else:
		images, labels = mndata.load_training()
		X,Y=norm(images, do_normalize),labels
		#X,Y=images,labels
		one = np.ones((len(X),1))
		X=np.concatenate((one, X), axis=1)
		theta=np.zeros(((np.shape(X)[1])))

		thetas=[]
		for x in xrange(10):
			print "Training for digit [",x,"]"
			YS=[]
			for c in xrange(len(Y)):
				if Y[c]==x:
					#print c, Y[c], x,1
					YS.append(1.0)
				else:
					#print c, Y[c], x,0
				 	YS.append(0.0)
			YS=np.array(YS)
			#train the model for 0 to 9
			print "Counting", x," in YS = [", collections.Counter(YS),"]"
			#gradient_theta, thetaxx, costs=gradient_descent(X ,Y,theta, logistic_loss, rounds=10, alpha=0.000000001, granularity=5, lamb=0.0)
			optimal_theta, res =logistic_descent_optimal(X,YS, theta, method="TNC", lamb=lamb)
			print "Cost [{}] Iterations [{}] Status [{}] Success [{}]".format(res.fun, res.nfev, res.status, res.success)
			thetas.append(optimal_theta)

		pickle.dump( thetas, open(theta_file, "wb"))
	return thetas

def train_digits2(retrain=False, theta_file="mnist_thetas.pickle", do_normalize=False, lamb=1.0):
	if os.path.isfile(theta_file) and not retrain:
		#print "File {} exists. Loading thetas".format(theta_file)
		thetas = pickle.load( open( theta_file, "rb" ) )
	else:
		images, labels = mndata.load_training()
		X,Y=norm(images, do_normalize),labels
		#X,Y=images,labels
		one = np.ones((len(X),1))
		X=np.concatenate((one, X), axis=1)
		theta=np.zeros(((np.shape(X)[1])))

		thetas=[]
		for x in xrange(10):
			print "Training for digit [",x,"]"
			YS=[]
			for c in xrange(len(Y)):
				if Y[c]==x:
					#print c, Y[c], x,1
					YS.append(1)
				else:
					#print c, Y[c], x,0
				 	YS.append(0)
			YS=np.array(YS)
			#train the model for 0 to 9
			print "Counting", x," in YS = [", collections.Counter(YS),"]"
			#gradient_theta, thetaxx, costs=gradient_descent(X ,Y,theta, logistic_loss, rounds=10, alpha=0.000000001, granularity=5, lamb=0.0)
			optimal_theta, res =logistic_descent_optimal2(X,YS, theta, method="TNC", lamb=lamb)
			print "Cost [{}] Iterations [{}] Status [{}] Success [{}]".format(res.fun, res.nfev, res.status, res.success)
			thetas.append(optimal_theta)

		pickle.dump( thetas, open(theta_file, "wb"))
	return thetas

def test(thetas, filename=""):
	test_images, test_labels=mndata.load_testing()
	ones = np.ones((len(test_images),1))
	test_images=np.concatenate((ones, test_images), axis=1)

	ok=[]
	error=[]

	for c in xrange(len(test_images)):
		test_image=test_images[c]
		test_label=test_labels[c]
		
		probs=[]
		for cc in range(len(thetas)):
			
			probs.append((cc,predict(test_image, thetas[cc])))
		#print "Expected digit: [",test_label,"]"
		#for di, prob in probs:
			#print "{:d}={:06.1f}%".format(di, round(prob*100,1))
		di,pr=max(probs,key=lambda item:item[1])
		#print (test_label,di,round(pr*100,1))

		if test_label != di:
			error.append((test_label,di,round(pr*100,1)))
			#print "Error"
			#print (test_label,di,round(pr*100,1))
		else:
			ok.append((test_label,di,round(pr*100,1)))
			
	print "Result for [{:50}] : OK: {:4d} Error: {:4d}".format(filename, len(ok), len(error))


do_normalize=False
filename="optimal.lambda00.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.0)
test(thetas, filename)

filename="optimal.lambda01.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.1)
test(thetas, filename)

filename="optimal.lambda02.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.2)
test(thetas, filename)

filename="optimal.lambda03.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.3)
test(thetas, filename)

filename="optimal.lambda04.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.4)
test(thetas, filename)

filename="optimal.lambda05.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.5)
test(thetas, filename)

filename="optimal.lambda06.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.6)
test(thetas, filename)

filename="optimal.lambda07.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.7)
test(thetas, filename)

filename="optimal.lambda08.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.8)
test(thetas, filename)

filename="optimal.lambda09.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.9)
test(thetas, filename)

filename="optimal.lambda10.mnist_thetas.pickle"
thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=1.0)
test(thetas, filename)


#do_normalize=True
#
#filename="normalized.optimal.lambda0.mnist_thetas.pickle"
#thetas=train_digits(retrain=False, theta_file=filename,do_normalize=do_normalize, lamb=.0)
#test(thetas, filename)



exit()



for x in range(10):
	label = labels[x]
	pixels = images[x][:]
	pixels = np.array(pixels, dtype='uint8')
	pixels = pixels.reshape((28, 28))
	plt.title('Label is {label}'.format(label=label))
	plt.imshow(pixels, cmap='gray')
	plt.show()
