import numpy as np # for linear algebra
import matplotlib.pyplot as plt # for plotting

def sigmoid(x):
    # Initialize our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # f'(x) = e^(-x) / (1 + e^(-x))^2 = f(x) * (1 - f(x)), which is more efficient.
    return sigmoid(x) * (1 - sigmoid(x))

# Assume male -> 0, female -> 1

# Mean Squared Error Loss Function, which is commonly used for regression tasks.
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred)**2).mean()

# Example usage:
y_true = np.array([1, 0, 0, 1]) # This is the ground truth (correct) values.
y_pred = np.array([0, 0, 0, 0]) # These are the predicted values, our model (currently) predicts 0 for all.

print(mse_loss(y_true, y_pred)) # 0.5

# Define a single neuron
class Neuron:
    def __init__(self,weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        return sigmoid(np.dot(self.weights, inputs) + self.bias)  # np.dot is for dot product
    

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
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1) # PS: the function of feedforward is from Neurons'
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset
        loss_history = []

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * sigmoid_derivative(sum_o1)
                d_ypred_d_w6 = h2 * sigmoid_derivative(sum_o1)
                d_ypred_d_b3 = sigmoid_derivative(sum_o1)

                d_ypred_d_h1 = self.w5 * sigmoid_derivative(sum_o1)
                d_ypred_d_h2 = self.w6 * sigmoid_derivative(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * sigmoid_derivative(sum_h1)
                d_h1_d_w2 = x[1] * sigmoid_derivative(sum_h1)
                d_h1_d_b1 = sigmoid_derivative(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * sigmoid_derivative(sum_h2)
                d_h2_d_w4 = x[1] * sigmoid_derivative(sum_h2)
                d_h2_d_b2 = sigmoid_derivative(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(all_y_trues, y_preds)
            loss_history.append(loss)

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
        return loss_history
            
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()  
loss_history = network.train(data, all_y_trues)

# Plot the loss history
plt.plot(loss_history)
plt.title("Neural Network Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig("images/loss.png")  # Save the figure
plt.show()

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.967 -> Female
print("Frank: %.3f" % network.feedforward(frank)) # 0.056 -> Male