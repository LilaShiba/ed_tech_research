import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


class DeepNetRunner:
    def __init__(self, name: str, layer_sizes: List[int], net_type: int) -> None:
        """
        Initialize the neural network.

        Parameters:
        - name: Name of the neural network.
        - layer_sizes: A list containing the sizes of each layer in the network.
        - net_type: An integer representing the type of network.
        """
        self.name: str = name
        self.layer_sizes: List[int] = layer_sizes
        self.type: str = self._get_net_type(net_type)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.create_network()

    def _get_net_type(self, net_type: int) -> str:
        """
        Retrieve the type of network based on the provided integer.

        Parameters:
        - net_type: An integer representing the type of network.

        Returns:
        - A string representing the type of network.
        """
        types = {
            1: 'cot',
            2: 'matrix',
            3: 'corpus',
            4: 'snd'
        }
        return types.get(net_type, 'unknown')

    def create_network(self) -> None:
        """
        Initialize the weights and biases for each layer in the network.
        """
        # Loop through each layer and initialize weights and biases
        for i in range(len(self.layer_sizes) - 1):
            # Weights are initialized with random values
            self.weights.append(np.random.randn(
                self.layer_sizes[i + 1], self.layer_sizes[i]))
            # Biases are initialized with zeros
            self.biases.append(np.zeros((self.layer_sizes[i + 1], 1)))

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters:
        - z: Weighted sum matrix.

        Returns:
        - Activation matrix.
        """
        # The sigmoid function maps any value to a value between 0 and 1
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid activation function.

        Parameters:
        - z: Weighted sum matrix.

        Returns:
        - Derivative matrix.
        """
        # Derivative of sigmoid function is used during backpropagation step
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the network.

        Parameters:
        - X: Input matrix.

        Returns:
        - Output matrix after forward pass.
        """
        self.a = [X]  # Store the input matrix
        self.z = []   # Initialize an empty list to store weighted sums

        # Loop through each layer, calculate weighted sum and activation
        for w, b in zip(self.weights, self.biases):
            self.z.append(np.dot(w, self.a[-1]) + b)  # Weighted sum
            self.a.append(self.sigmoid(self.z[-1]))   # Activation

        return self.a[-1]  # Return the final output

    def backpropagate(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """
        Perform backpropagation and update weights and biases.

        Parameters:
        - X: Input matrix.
        - y: Target output matrix.
        - learning_rate: Learning rate for weight and bias updates.
        """
        m = X.shape[1]  # Number of training examples

        # Compute the error in the output layer
        dz = [self.a[-1] - y]
        dw = [np.dot(dz[0], self.a[-2].T) / m]
        db = [np.sum(dz[0], axis=1, keepdims=True) / m]

        # Compute the error for the rest of the layers
        for i in range(2, len(self.layer_sizes)):
            dz.append(np.dot(self.weights[-i + 1].T, dz[-1])
                      * self.sigmoid_derivative(self.z[-i]))
            dw.append(np.dot(dz[-1], self.a[-i-1].T) / m)
            db.append(np.sum(dz[-1], axis=1, keepdims=True) / m)

        # Reverse the computed gradients to match the order of layers
        dw = dw[::-1]
        db = db[::-1]

        # Update the weights and biases
        self.weights = [w - learning_rate *
                        dw_i for w, dw_i in zip(self.weights, dw)]
        self.biases = [b - learning_rate *
                       db_i for b, db_i in zip(self.biases, db)]

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the Mean Squared Error loss.

        Parameters:
        - y_pred: Predicted output matrix.
        - y_true: True output matrix.

        Returns:
        - Loss value.
        """
        m = y_true.shape[1]
        return (1/(2*m)) * np.sum(np.square(y_pred - y_true))

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Train the neural network for a specified number of epochs.

        Parameters:
        - X: Input matrix.
        - y: Target output matrix.
        - epochs: Number of training epochs.
        - learning_rate: Learning rate for weight and bias updates.

        Returns:
        - Dictionary containing loss and parameters at each epoch.
        """
        output_dict = {}

        for epoch in range(epochs):
            y_pred = self.feedforward(X)
            loss = self.compute_loss(y_pred, y)
            self.backpropagate(X, y, learning_rate)

            output_dict[epoch] = {
                "loss": loss,
                "weights": [w.copy() for w in self.weights],
                "biases": [b.copy() for b in self.biases]
            }

        return output_dict

    def plot_training_output(self, output_dict: Dict[int, Dict[str, np.ndarray]]) -> None:
        """
        Plot the training loss over epochs using data from output_dict.

        Parameters:
        - output_dict: Dictionary containing loss and parameters at each epoch.
        """
        epochs = list(output_dict.keys())
        loss_values = [output_dict[epoch]['loss'] for epoch in epochs]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b')
        plt.title(f"Training Loss over Epochs ({self.name})")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    mu = 0          # mean
    sigma = 1       # standard deviation
    input_size = 10  # for example
    train_size = 1000  # number of training examples
    out_put_size = 3

    # Using normal distribution
    X_train = np.random.normal(mu, sigma, (input_size, train_size))
    # 1 output, 1000 examples
    y_train = np.random.normal(mu, sigma, (out_put_size, train_size))

    # Initialize a deep network: [10, 5, 5, 1] means an input layer of size 10,
    # two hidden layers of size 5, and an output layer of size 1
    deep_network = DeepNetRunner(name="DeepNetwork", layer_sizes=[
                                 input_size, 5, 5, 5, 5, 5, 5, 5, 5, out_put_size], net_type=3)

    # Train the deep network
    output_dict_deep = deep_network.train(
        X_train, y_train, epochs=12, learning_rate=0.01)

    # Plot the training loss for the deep network
    deep_network.plot_training_output(output_dict_deep)

    # Initialize a wide network: [10, 100, 1] means an input layer of size 10,
    # a hidden layer of size 100, and an output layer of size 1
    wide_network = DeepNetRunner(name="WideNetwork", layer_sizes=[
                                 input_size, 10000, 10000, out_put_size], net_type=2)

    # Train the wide network
    output_dict_wide = wide_network.train(
        X_train, y_train, epochs=12, learning_rate=0.01)

    # Plot the training loss for the wide network
    wide_network.plot_training_output(output_dict_wide)

    # Deep wide

    deep_wide_network = DeepNetRunner(name="DeepWideNetwork", layer_sizes=[
        input_size, 10000, 10000, 10000, 10000, 10000, 10000, 10000, out_put_size], net_type=2)

    # Train the wide network
    output_dict_wide = deep_wide_network.train(
        X_train, y_train, epochs=12, learning_rate=0.01)

    # Plot the training loss for the wide network
    deep_wide_network.plot_training_output(output_dict_wide)
