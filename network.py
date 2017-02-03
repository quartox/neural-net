import numpy as np

class Network:
    def __init__(self, num_predictors, num_neurons):
        """Creates the layers of neurons in a neural network.

        Args:
          num_predictors - the number of predictors
          num_neurons - a list of the number of neurons in the hidden and output
              layers of the network.
        """
        self.num_predictors = num_predictors
        self.num_layers = len(num_neurons)
        self.layers = [Layer(self.num_predictors + 1, num_neurons[0])]
        for i_layer in range(1, self.num_layers):
            num_inputs = len(self.layers[-1].outputs)
            if i_layer < self.num_layers-1:
                self.layers.append(Layer(num_inputs, num_neurons[i_layer]))
            else:
                self.layers.append(Layer(num_inputs, num_neurons[i_layer],
                                         is_output=True))

    def feedforward(self, x):
        """Given the input record x, computes the output of the neural network.
        Adds the bias neuron to each hidden layer.

        Args:
          x - an input record (list-like).

        Returns:
          A list of numpy arrays of outputs
        """
        outputs = []
        inputs = np.ones(self.num_predictors+1)
        inputs[1:] = x
        for layer in self.layers:
            outputs.append(layer.compute_outputs(inputs))
            inputs = outputs[-1]
        return outputs

    def backprop(self, x, y):
        outputs = self.feedforward(x)
        deltas = []
        deltas.append((2 * (self.layers[-1].outputs - y) *
                       self.activation_derivative(self.layers[-1].outputs)))
        for i_layer in range(self.num_layers-2, -1, -1):
            num_neurons = self.layers[i_layer].num_neurons
            weighted_deltas = np.zeros(self.layers[i_layer].num_neurons)
            for i_neuron in range(num_neurons):
                weighted_deltas[i_neuron] = np.dot(
                    self.layers[i_layer+1].weights[i_neuron+1,:], deltas[-1])
            activation_deriv = self.activation_derivative(outputs[i_layer][1:])
            deltas.append(np.multiply(activation_deriv, weighted_deltas))
        return deltas

    @staticmethod
    def activation_derivative(x):
        return 1 - np.square(x)

class Layer:
    def __init__(self, num_inputs, num_neurons, is_output=False):
        """Creates a layer of neurons in a neural network.

        Args:
          num_inputs - the number of inputs to each neuron in this layer.
          num_neurons - the number of neurons in this layer.
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = np.random.randn(self.num_inputs, self.num_neurons)
        self.is_output = is_output
        if self.is_output:
            self.outputs = np.empty(num_neurons)
        else:
            self.outputs = np.ones(num_neurons+1)
        self.deltas = np.empty(num_neurons)

    def compute_outputs(self, inputs):
        """Computes the output of each neuron in this layer.

        Args:
          inputs - the inputs from the previous layer.

        Returns:
          the outputs of the neurons in this layer, a numpy.array
        """
        if self.is_output:
            self.outputs = np.tanh(np.dot(inputs, self.weights))
        else:
            self.outputs[1:] = np.tanh(np.dot(inputs, self.weights))
        return self.outputs
