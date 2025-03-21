import numpy as np 

class Softmax: 
    #standard fully connected layer with softmax activation

    def __init__(self, input_len, nodes):
        #divide by input_len to reduce variance
        self.weights = np.random.randn(input_len, nodes) /input_len #random weights
        self.biases = np.zeros(nodes) #bias vector initialized to zero

    def forward (self, input):

        self.last_input_shape = input.shape #input shape before flattening

        #perform forward pass

        input = input.flatten() #flatten to one dimension

        self.last_input = input #input shape after flattening

        input_len, nodes = self.weights.shape #get dimensions of weight matrix: input_len is number of input features and nodes is number of output neurons

        totals = np.dot(input, self.weights)+self.biases #compute weighted sum
        self.last_totals = totals
        exp = np.exp(totals) #calculate exponentials for softmax
        return exp/np.sum(exp, axis = 0) #apply softmax normalization
    
    def backprop(self, d_L_d_out, learn_rate):
 
        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)

