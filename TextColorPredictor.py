
'''
https://thecodacus.com/neural-network-scratch-python-no-libraries/#.WsgUsIjwZM0
'''
import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.1
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        #return 1 / (1 + math.exp(-x * 1.0))
        return 1/(1+np.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0


class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword()

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output
   
class Showoutput:

    def __init__(self, bgColor,textColor):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.add_patch(
            patches.Rectangle(
                (0.1, 0.1),   # (x,y)
                1,          # width
                1,          # height
                facecolor=self.rgbtohax(bgColor)
            )
        )
        ax1.text(0.4, 0.6, 'White' if textColor[0]==0 else 'Black',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color=self.rgbtohax([255,255,255]) if textColor[0]==0 else self.rgbtohax([0,0,0]),
        fontsize=15)

        fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')
        #fig1.show()
        img=mpimg.imread('rect1.png')
        imgplot = plt.imshow(img)
        plt.show()

    def rgbtohax(self,rgbcolor):
        return '#%02x%02x%02x' % (rgbcolor[0], rgbcolor[1], rgbcolor[2])
    
def main():
    topology = []
    #parth topology.append(2)
    #parthtopology.append(3)
    #parthtopology.append(2)

    topology.append(3)
    topology.append(3)
    topology.append(1)

    net = Network(topology)
    #Neuron.eta = 0.1
    #Neuron.alpha = 0.05
    Neuron.eta = 1.0
    Neuron.alpha = 0.09
    errArr = []
    #while True:
    for i in range(2500):
        err = 0
        
        my_data = genfromtxt('Input.csv', delimiter=',',skip_header=1)
        inputs = my_data[0:200, [0,1,2]]
        inputs = inputs/255 # maximum of X array (scale input)
        outputs = my_data[0:200, [3]]
        
        
        #inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        #outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForword()
            net.backPropagate(outputs[i])
            err = err + net.getError(outputs[i])
        print("error: "+ str(err))
        errArr.append(err)
    #     if err < 0.01:
    #         break
    


    plt.plot(errArr,'r')
    plt.show()

    #crossvalidation start
    inputs = my_data[200:, [0,1,2]] #getting crossvalidation set
    inputs = inputs/255 # maximum of X array (scale input)
    outputs = my_data[200:, [3]] 
    crossvalidationOutput = [] #define arry for write in file
    outputcolorArr = []
    for i in range(len(inputs)): #loop for the data we have
        net.setInput(inputs[i]) 
        net.feedForword()
        outputColor=net.getThResults()
        outputcolorArr.append(outputColor)
        #crossvalidationOutput.append([inputs[i][0],inputs[i][1],inputs[i][2],'-->',outputColor, 'Correct' if outputs[i][0]==outputColor else 'Wrong'])
        #multiply input back to descale 
        crossvalidationOutput.append([int(inputs[i][0]*255),int(inputs[i][1]*255),int(inputs[i][2]*255),outputColor[0], outputs[i][0]])
    
    #Save in csv file
    np.savetxt("CrossValidation.csv", crossvalidationOutput,fmt="%d",delimiter=",")
    
    notmatched = 0
    for i in range(len(outputs)):
        if int(outputs[i][0]) != int(outputcolorArr[i][0]):
            notmatched = notmatched+1 

    print("Total notmatched count#:" + str(notmatched))
    #crossvalidation end

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        c = input("type 3nd input :")
        inputColor = [int(a), int(b), int(c)]
        net.setInput(inputColor)
        net.feedForword()
        outputColor=net.getThResults()
        print(outputColor)
        output = Showoutput(inputColor,outputColor)


if __name__ == '__main__':
    main()