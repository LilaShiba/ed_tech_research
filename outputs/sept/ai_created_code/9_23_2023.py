import numpy as np


class TextNeuralNetwork:
    '''
    Agent COT for neural net build
    '''

    def __init__(self, input_size, hidden_size, output_size, use_backpropagation=False, use_recurrent=False):
        '''
        Initialize the neural network with the given input, hidden, and output sizes
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Set the flags for using backpropagation and/or recurrent pipelines
        self.use_backpropagation = use_backpropagation
        self.use_recurrent = use_recurrent
        # Initialize the weights and biases
        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        if self.use_recurrent:
            self.weights_hidden_hidden = np.random.randn(
                self.hidden_size, self.hidden_size)
            self.weights_hidden_output = np.random.randn(
                self.hidden_size, self.output_size)
            self.bias_output = np.zeros((1, self.output_size))

    def forward(self, input_data):
        '''
        Perform the forward pass through the neural network
        '''
        self.hidden_activation = np.dot(
            input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)
        if self.use_recurrent:
            self.recurrent_activation = np.dot(
                self.hidden_output, self.weights_hidden_hidden)
            self.hidden_output = self.sigmoid(
                self.hidden_activation + self.recurrent_activation)
            self.output_activation = np.dot(
                self.hidden_output, self.weights_hidden_output) + self.bias_output
            self.output = self.sigmoid(self.output_activation)
            return self.output

    def backward(self, input_data, target, learning_rate):
        '''
        Perform the backpropagation algorithm to update the weights and biases
        Calculate the gradients of the weights and biases
        '''
        output_error = target - self.output
        output_delta = output_error * \
            self.sigmoid_derivative(self.output_activation)
        hidden_output_error = np.dot(
            output_delta, self.weights_hidden_output.T)
        hidden_output_delta = hidden_output_error * \
            self.sigmoid_derivative(self.hidden_activation)
        if self.use_recurrent:
            recurrent_output_error = np.dot(
                hidden_output_delta, self.weights_hidden_hidden.T)
            recurrent_output_delta = recurrent_output_error * \
                self.sigmoid_derivative(self.hidden_activation)
            # Update the weights and biases
            self.weights_hidden_output += learning_rate * \
                np.dot(self.hidden_output.T, output_delta)
            self.bias_output += learning_rate * \
                np.sum(output_delta, axis=0, keepdims=True)
            self.weights_input_hidden += learning_rate * \
                np.dot(input_data.T, hidden_output_delta)
            self.bias_hidden += learning_rate * \
                np.sum(hidden_output_delta, axis=0, keepdims=True)

            self.weights_hidden_hidden += learning_rate * \
                np.dot(self.hidden_output.T, recurrent_output_delta)

    def sigmoid(self, x):
        '''
        Returns: Sigmoid activation function
        '''
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        '''
        Returns: Derivative of the sigmoid function

        Please note that this is a simplified implementation and 
        there might be additional steps and considerations needed depending 
        on the specific requirements of your text understanding task.

        '''
        return self.sigmoid(x) * (1 - self.sigmoid(x))
