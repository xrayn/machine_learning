import scipy.io
import numpy as np
import sys

sys.path.append('../logistic_regression')
from logistic_regression_lib import *


mat = scipy.io.loadmat('data/ex3data1.mat')

X=mat["X"]
y=mat["y"]

print( mat["X"].shape)
print(X[0])
print( mat["y"].shape)
print(y[0])
matheta = scipy.io.loadmat('data/ex3weights.mat')

theta1= matheta["Theta1"]
theta2= matheta["Theta2"]


class NeuronalNet:
	def __init__(self, x, w1, w2, y):
		self.input_layer=x
		self.output_layer=[]
		self.hidden_layer=None
		self.y=y
		self.w1=w1
		self.w2=w2
		self.m, self.n = self.input_layer.shape
		self.cost=0

		# the network
		#
		#  input    w1        hidden
		#   0 ----- w1_0 --- (hidden) --- w2_0 --- (output)
		#   0 ----- w1_1 --- (hidden) --- w2_1 --- (output)
		#   0 ----- w1_2 --- (hidden) --- w2_2 --- (output)

		#self.hidden_layer=sigmoid(self.input_layer.T.dot(w1.T))
		#self.hidden_layer=self.add_1s_hidden_layers(self.hidden_layer)
		#self.output_layer=sigmoid(self.hidden_layer.T.dot(w1.T))
	def derivate(self, x):
		return x*(1-x)
	def logistic_cost(self,h_x,Y):
		"""
		Calculates the cost of the current values
		"""
		m,n=self.input_layer.shape
		
		h_x_sum=np.dot(-Y,np.log(h_x))-np.dot((1.0-Y),np.log(1.0-h_x))
		

		#regularized theta does not consider the thetha[0]	
		#thetaR = theta[1:]
		#lambda_term=((lamb)/2.0*m)*thetaR.dot(thetaR)
		#h_x_sum=h_x_sum+lambda_term
		cost=(float(1.0)/(m))*h_x_sum
		self.cost+=cost
		return cost
	
	def logistic_cost_regularized(self,h_x, lamb=1):
		
		theta_1_sum=0
		theta_2_sum=0
		# this executes the sum for 25*400		
		for i in range(self.w1.shape[0]):
			for j in range(self.w1.shape[1]):
				thetaR1 = self.w1[i][j]
				theta_1_sum+=thetaR1*thetaR1
		
		for i in range(self.w2.shape[0]):
			for j in range(self.w2.shape[1]):
				thetaR2 = theta2[i][j]
				theta_2_sum+=thetaR2*thetaR2

		# this executes the sum for 10*25
			
		lambda_term=((lamb)/(2.0*self.m))* (theta_1_sum+theta_2_sum)
		print((lamb)/(2.0*self.m))
		print((theta_1_sum+theta_2_sum))
		print(lambda_term)
		h_x_sum=h_x+lambda_term
		self.cost_regularized=h_x_sum
		return h_x_sum

	def feed_forward(self, index=0):
		#print("-------------")
		#print(self.w1.shape)
		#print(self.w1)
		#print("-------------")
		#print(self.w1.shape)
		#print(self.w1.T)
		#print("-------------")
		self.hidden_layer=sigmoid(self.input_layer[index].dot(self.w1.T))
		self.hidden_layer=add_1s_hidden_layers(self.hidden_layer)
		self.output_layer=sigmoid(self.hidden_layer.T.dot(self.w2.T))
		


		#first transform y to a vector
		
		y_t=np.zeros(10)
		#y_t[self.y-1]=1.0
		
		y_t[self.y[index]-1]=1.
		#print(y_t)
		m,n=self.input_layer.shape
		cost=self.logistic_cost(self.output_layer, y_t)
		res=(np.argmax(self.output_layer), np.max(self.output_layer))
		
		return res
	def get_cost(self):
		return self.cost

	def get_cost_regularized(self):
		return self.cost+self.cost_regularized
	
	def add_1s(self,X):
		m,n=X.shape
		one = np.ones((m,1))
		# this simply adds 1's in front of the X
		X=np.concatenate((one, X), axis=1)
		return X	
	def add_1s_hidden_layers(layer):
		return np.insert(layer, 0, 1., axis=0)

def add_1s(X):
	m,n=X.shape
	one = np.ones((m,1))
	# this simply adds 1's in front of the X
	X=np.concatenate((one, X), axis=1)
	return X	


def add_1s_hidden_layers(layer):
	return np.insert(layer, 0, 1., axis=0)

def predict(x, theta1, theta2):
	#print(x[0,:].shape)
	#print("theta1", theta1.shape)
	out_layer=propagate(x, theta1, theta2)
	return np.argmax(out_layer), np.max(out_layer)


def propagate(x, theta1, theta2):
	l1=sigmoid(x.T.dot(theta1.T))
	l1_1s=add_1s_hidden_layers(l1)
	l2=sigmoid(l1_1s.T.dot(theta2.T))
	return l2

def predict_floop(x, theta1, theta2):
	#print(x[0,:].shape)
	#print("theta1", theta1.shape)
	
	l1=np.zeros((theta1.shape[0],1))
	for i in range(theta1.shape[0]):
		l1[i]=x.dot(theta1[i,:])
	
	l1=sigmoid(l1)
	
	one = np.ones((len(l1),1))
	#l1=np.concatenate((one, l1), axis=1)	
	l2=np.zeros((theta2.shape[0],1))
	l1_1s=np.insert(l1, 0, 1., axis=0)
	for i in range(theta2.shape[0]):
		l2[i]=l1_1s.T.dot(theta2[i,:])

	l2=sigmoid(l2)
	
	return np.argmax(l2), np.max(l2)



# this simply adds 1's in front of the X
X=add_1s(X)
print()
print(theta1.shape)
w1_neurons=25
w1=np.random.rand(w1_neurons,X.shape[1]) 
print(w1.shape)
w2_neurons=10
print(theta2.shape)
w2=np.random.rand(w2_neurons,w1.shape[0]+1) 
print(w2.shape)




nn = NeuronalNet(X, theta1, theta2, y)
#nn = NeuronalNet(X, w1, w2, y)
cost=0.0

for i in range(X.shape[0]):
	index, value= nn.feed_forward(i)
	
	if not y[i]-1 == index:
		print(y[i]-1, index, value, "ERROR")
	else:
		print(y[i]-1, index, value, "OK")

nn.logistic_cost_regularized(cost, 1)

cost=nn.get_cost()
print("Cost ["+str(cost)+"]")

cost_regularized=nn.get_cost_regularized()
print("Cost regularized["+str(cost_regularized)+"]")	


#ok=0
#err=0
#
#for j in range(X.shape[0]):
#	idx, value = predict(X[j,:],theta1,theta2)
	#
#	if (idx+1) == y[j]:
#		ok=ok+1.0
#	else:
#	 	err=err+1.0
#print( (ok/5000*100), (err/5000*100))

exit()