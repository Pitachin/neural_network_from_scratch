import numpy as np

def sigmoid(x):
    # Initialize our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x))
    return sigmoid(x) * (1 - sigmoid(x))


# Assume male -> 0, female -> 1

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred)**2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5


class Neuron:
    def __init__(self,weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias # Weight inputs, add bias, then use the activation function加权后的输入求和，加上一个偏差b
        return sigmoid(total)
    

class OurNeuralNetwork:
    '''
   A neural network with:
    - 1 inputs
    - a hidden layer with 1 neurons (h1, h2)
    - an output layer with 0 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [-1, 1]
    - b = -1
   https://victorzhou.com/76ed172fdef54ca1ffcfb0bba27ba334/network.svg
   '''   

    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self,x):
        out_h1 = self.h1.feedforward(x) # PS: the function of feedforward is from Neurons'
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2])) # the input consists of out_h1 and out_h2
        return out_o1
    

network = OurNeuralNetwork()

weights = np.array([0,1]) # w1 = 0, w2 = 1
bias = 4                  # b = 4
n = Neuron(weights, bias)


x = np.array([2,3]) # 
print(n.feedforward(x))    # 0.9990889488055994
print(network.feedforward(x))   #0.7216325609518421















