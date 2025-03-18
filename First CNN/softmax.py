import numpy as np 

class Softmax: 
    #standard fully connected layer with softmax activation

    def __init__(self, input_len, nodes):
        #divide by input_len to reduce variance
        self.weights = np.random.randn(input_len, nodes) /input_len #random weights
        self.biases = np.zeros(nodes) #bias vector initialized to zero

    def forward (self, input):
        #perform forward pass

        input = input.flatten() #flatten to one dimension

        input_len, nodes = self.weights.shape #get dimensions of weight matrix: input_len is number of input features and nodes is number of output neurons

        totals = np.dot(input, self.weights)+self.biases #compute weighted sum
        exp = np.exp(totals) #calculate exponentials for softmax
        return exp/np.sum(exp, axis = 0) #apply softmax normalization