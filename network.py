import numpy as np

class Network:
    def __init__(self, num_inputs, num_neurons):
        """Creates the layers of neurons in a neural network.

        Args:
          num_inputs - the number of inputs for the first layer of the network.
          num_neurons - a list of the number of neurons in the hidden and output
              layers of the network.
        """
        self.layers = [Layer(num_neurons[0], num_inputs)]
        for layer_index in range(1, len(num_neurons)):
            current_layer_num_inputs = self.layers[-1].num_neurons + 1
            self.layers.append(
                Layer(current_layer_num_inputs, num_neurons[layer_index]))

    def feedforward(self, x):
        """Given the input record x, computes the output of the neural network.
        Adds the bias neuron to each hidden layer.

        Args:
          x - an input record (list-like).

        Returns:
          the output of the network (a float or numpy.array of floats).
        """
        current_layer_inputs = x
        for layer in self.layers:
            current_layer_outputs = layer.compute_outputs(current_layer_inputs)
            current_layer_inputs = np.insert(current_layer_outputs, 0, 1)
        if len(current_layer_outputs) == 1:
            return current_layer_outputs[0]
        else:
            return current_layer_outputs

class Layer:
    def __init__(self, num_inputs, num_neurons):
        """Creates a layer of neurons in a neural network. Sets the seed
        to ensure that no two neurons are initialized with the same weights/bias.

        Args:
          num_inputs - the number of inputs to each neuron in this layer.
          num_neurons - the number of neurons in this layer.
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.neurons = []
        for i in range(self.num_neurons):
            np.random.seed(i)
            self.neurons.append(Neuron(self.num_inputs))

    def compute_outputs(self, inputs):
        """Computes the output of each neuron in this layer.

        Args:
          inputs - the inputs from the previous layer.

        Returns:
          the outputs of the neurons in this layer, a numpy.array
        """
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.compute_output(inputs))
        return np.array(outputs)

class Neuron:
    def __init__(self, num_inputs):
        """Creates a neuron."""
        self.num_inputs = num_inputs
        self.weights = np.random.randn(self.num_inputs)
        self.bias = np.random.randn()

    def compute_output(self, inputs):
        return self.sigmoid(np.inner(self.weights, inputs) + self.bias)

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
