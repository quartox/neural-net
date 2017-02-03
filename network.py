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
        self.layers = [HiddenLayer(self.num_predictors + 1, num_neurons[0])]
        for i_layer in range(1, self.num_layers):
            num_inputs = len(self.layers[-1].outputs)
            if i_layer < self.num_layers-1:
                self.layers.append(HiddenLayer(num_inputs, num_neurons[i_layer]))
            else:
                self.layers.append(OutputLayer(num_inputs, num_neurons[i_layer]))

    def feedforward(self, x):
        """Given the input record x, computes the output of the neural network.
        Adds the bias neuron to each hidden layer.

        Args:
          x - an input record (list-like).

        Returns:
          A list of numpy arrays of outputs from all layers.
        """
        outputs = []
        inputs = np.ones(self.num_predictors+1)
        inputs[1:] = x
        for layer in self.layers:
            outputs.append(layer.compute_outputs(inputs))
            inputs = outputs[-1]
        return outputs

    def backprop(self, x, y):
        """Back propagation of the errors to all neurons in the network.

        Args:
          x - an input record (list-like).
          y - the response for the record.

        Returns:
          A list of deltas (errors) from each layer.
        """
        self.feedforward(x)
        deltas = []
        deltas.append(self.layers[-1].compute_deltas(y))
        for i_layer in range(self.num_layers-2, -1, -1):
            weighted_deltas = self.compute_weighted_deltas(i_layer)
            deltas.append(self.layers[i_layer].compute_deltas(weighted_deltas))
        return deltas

    def compute_weighted_deltas(self, i_layer):
        """Computes the dot product of the weights and deltas from the next layer
        above. For one neuron in the current layer this is equivalent to the
        sum of the weights to each of the next layer neurons times the deltas
        for that neuron.
        """
        return np.dot(self.layers[i_layer+1].weights[1:,:],
                      self.layers[i_layer+1].deltas)


class HiddenLayer:
    def __init__(self, num_inputs, num_neurons):
        """Creates a layer of neurons in a neural network.

        Args:
          num_inputs - the number of inputs to each neuron in this layer.
          num_neurons - the number of neurons in this layer.
        """
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.weights = np.random.randn(self.num_inputs, self.num_neurons)
        self.outputs = np.ones(num_neurons+1)
        self.deltas = np.empty(num_neurons)

    def compute_outputs(self, inputs):
        """Computes the output of each neuron in this layer.

        Args:
          inputs - the inputs from the previous layer.

        Returns:
          the outputs of the neurons in this layer with a bias output
        """
        self.outputs[1:] = np.tanh(np.dot(inputs, self.weights))
        return self.outputs

    def compute_deltas(self, weighted_deltas):
        """Computes the deltas, which are the error with respect to the signal.
        For the hidden layers the deltas are the elementwise multiplication of
        the derivative of the activation function times the weighted deltas from
        the next layer above.

        Args:
          weighted_deltas - the dot product of the weights with next layer up deltas

        Returns:
          the delta(s) from the current hidden layer
        """
        activation_deriv = self.activation_derivative(self.outputs[1:])
        self.deltas = np.multiply(activation_deriv, weighted_deltas)
        return self.deltas

    @staticmethod
    def activation_derivative(x):
        """The tanh derivative. Note that this derivative technically is being
        evaluated at the signal: dot(inputs, weights), but it can be computed
        with the output: tanh(dot(inputs, weights))
        """
        return 1 - np.square(x)


class OutputLayer:
    def __init__(self, num_inputs, num_outputs):
        """Creates a layer of output neurons.

        Args:
          num_inputs - the number of inputs to each neuron in this layer.
          num_outputs - the number of outputs from the neural network.
        """
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.weights = np.random.randn(self.num_inputs, self.num_outputs)
        self.outputs = np.empty(self.num_outputs)
        self.deltas = np.empty(self.num_outputs)

    def compute_outputs(self, inputs):
        """Computes the output of each neuron in this layer.

        Args:
          inputs - the inputs from the previous layer.

        Returns:
          the outputs the neural network, a numpy.array
        """
        self.outputs = np.tanh(np.dot(inputs, self.weights))
        return self.outputs

    def compute_deltas(self, response):
        """Computes the deltas, which are the error with respect to the signal.

        Args:
          response - the true observed response of this record

        Returns:
          the delta(s) from the output layer
        """
        self.deltas = (2 * (self.outputs - response) *
                       self.activation_derivative(self.outputs))
        return self.deltas

    @staticmethod
    def activation_derivative(x):
        """The tanh derivative. Note that this derivative technically is being
        evaluated at the signal: dot(inputs, weights), but it can be computed
        with the output: tanh(dot(inputs, weights))
        """
        return 1 - np.square(x)
