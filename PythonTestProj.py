
import numpy as np
from numpy import genfromtxt
import time
import matplotlib.pyplot as plt

# X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input data for prediction)
#X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
#y = np.array(([92], [86], [89]), dtype=float)

#X = np.random.binomial(1, 0.5, (n_samples, n_in))
my_data = genfromtxt('Input.csv', delimiter=',',skip_header=1)
X = my_data[:, [0,1,2]]
#print(X)
#T = X ^ 1
y = my_data[:, [3]]
#print(y)

#xPredicted = np.array(([255, 211, 186]), dtype=float)
#x = [[79,37,66]] # answer should be white (0,1)
#x=[[255, 211, 186]] # answer should be black (1,0)

# scale units - existing code
    #X = X/np.amax(X, axis=0) # maximum of X array
    #xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
    #y = y/100 # max test score is 100

# for me - commented as i dont neeed to scale
#X = X/255 # maximum of X array
#xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
#y = y/100 # max test score is 100

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 3
    self.outputSize = 2
    self.hiddenSize = 3
    

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x2) weight matrix from hidden to output layer

    self.lossArr = []
    self.sigmoidArr = []

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    self.sigmoidArr.append(o);
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self,predictInput):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(predictInput))
    print("Output: \n" + str(self.forward(predictInput)))

NN = Neural_Network()

for i in range(500): # trains the NN 1,000 times
  #print("# " + str(i) + "\n")
  #print("Input (scaled): \n" + str(X))
  #print("Actual Output: \n" + str(y))
  #print("Predicted Output: \n" + str(NN.forward(X)))
  #print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  #print("\n")
  loss = np.mean(np.square(y - NN.forward(X)))
  NN.lossArr.append(loss)
  print('Iteration: {0}, Loss: {1}'.format(str(i),loss))
  NN.train(X, y)

plt.plot(NN.lossArr,'r')
#plt.plot(NN.sigmoidArr,'b')
plt.show()
NN.saveWeights()
NN.predict(np.array([255, 211, 186]))
NN.predict(np.array([79,37,66]))