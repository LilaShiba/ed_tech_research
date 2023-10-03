import numpy as np
import random


class FeedForwardNN:
    """
    A simple feedforward neural network class with unique activation functions for each node.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 use_backpropagation: bool = False, use_recurrent: bool = False) -> None:
        """
        Initialize the neural network with the given parameters.

        Parameters:
        - input_size (int): The number of input nodes.
        - hidden_size (int): The number of hidden nodes.
        - output_size (int): The number of output nodes.
        - use_backpropagation (bool): Whether to use backpropagation for training.
        - use_recurrent (bool): Whether to include recurrent connections.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_backpropagation = use_backpropagation
        self.use_recurrent = use_recurrent

        # Initialize weights randomly
        self.weights_ih = np.random.randn(input_size, hidden_size)
        self.weights_ho = np.random.randn(hidden_size, output_size)

        # Initialize recurrent weights if needed
        if self.use_recurrent:
            self.weights_recurrent = np.random.randn(hidden_size, hidden_size)

        # Define a set of activation functions and their derivatives
        self.activation_functions = [
            (self.sigmoid, self.sigmoid_derivative),
            (self.tanh, self.tanh_derivative)
        ]

        # Assign a random activation function to each node in the hidden and output layers
        self.hidden_activation_functions = random.choices(
            self.activation_functions, k=self.hidden_size)
        self.output_activation_functions = random.choices(
            self.activation_functions, k=self.output_size)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid activation function."""
        return x * (1 - x)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of the hyperbolic tangent activation function."""
        return 1 - np.tanh(x)**2

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.

        Parameters:
        - inputs (np.ndarray): The input data.

        Returns:
        - output (np.ndarray): The output of the network.
        """
        # Ensure inputs is 2-dimensional
        inputs = np.atleast_2d(inputs)

        # Calculate activations of the hidden layer
        hidden_activations = np.dot(inputs, self.weights_ih)
        # Apply the assigned activation functions to the hidden layer
        hidden_output = np.array([func(hidden_activations[:, i])
                                 for i, (func, _) in enumerate(self.hidden_activation_functions)]).T

        # Calculate activations of the output layer
        output_activations = np.dot(hidden_output, self.weights_ho)
        # Apply the assigned activation functions to the output layer
        output = np.array([func(output_activations[:, i]) for i, (func, _) in enumerate(
            self.output_activation_functions)]).T

        return output

    def backward(self, inputs: np.ndarray, target: np.ndarray) -> None:
        """
        Perform backward propagation through the network.

        Parameters:
        - inputs (np.ndarray): The input data.
        - target (np.ndarray): The target data.
        """
        # Ensure inputs is 2-dimensional
        inputs = np.atleast_2d(inputs)

    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int) -> None:
        """
        Train the neural network for a specified number of epochs.

        Parameters:
        - inputs (np.ndarray): The input data.
        - targets (np.ndarray): The target data.
        - epochs (int): The number of epochs to train for.
        """
        for epoch in range(epochs):
            for input_vector, target_vector in zip(inputs, targets):
                if self.use_backpropagation:
                    self.backward(input_vector, target_vector)
                else:
                    self.forward(input_vector)


if __name__ == "__main__":
    # Assume we have 10 images of size 8x8, and there are 3 categories (N=3)
    num_images = 10
    image_size = 8
    num_categories = 3

    # Generate synthetic image data
    images = np.random.rand(num_images, image_size, image_size)

    # Flatten each image into a 1D array
    flattened_images = images.reshape(num_images, -1)

    # Assume the following one-hot encoded labels for the images
    labels = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    # Define the neural network
    input_size = flattened_images.shape[1]
    hidden_size = 100  # You may need to tune this value
    output_size = num_categories

    nn = FeedForwardNN(input_size, hidden_size, output_size,
                       use_backpropagation=True)

    # Train the neural network
    epochs = 20
    nn.train(flattened_images, labels, epochs)

    # Now you can use the nn.forward method to make predictions on unseen data
    # Assume 2 new unseen images
    new_images = np.random.rand(2, image_size, image_size)
    flattened_new_images = new_images.reshape(2, -1)
    predictions = nn.forward(flattened_new_images)

    # Convert the output of the neural network to class indices
    predicted_indices = np.argmax(predictions, axis=1)

    # Create a list of category names
    category_names = ["Category A", "Category B", "Category C"]

    # Map the predicted indices to category names
    predicted_categories = [category_names[index]
                            for index in predicted_indices]

    # Output the predicted categories for the new images
    print(predicted_categories)
