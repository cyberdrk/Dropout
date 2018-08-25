import numpy
import scipy.special
class NeuralNet(object):

    def __init__(self, n_input, n_hidden, n_output, learning_rate):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.learning_rate = learning_rate

        self.wih = numpy.random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, self.n_input))
        self.who = numpy.random.normal(0.0, pow(self.n_output, -0.5), (self.n_output, self.n_hidden))

    def train(self, inputs_list, targets_list):
        p = 0.5 # Dropout Error Rate
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        d1 = (numpy.random.rand(*hidden_inputs.shape) < p) / p
        hidden_inputs *= d1
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        d2 = (numpy.random.rand(*final_inputs.shape) < p) / p
        final_inputs *= d2
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
        self.wih += self.learning_rate * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def activation_function(self, x):
        return scipy.special.expit(x)

"""
Example usage of NeuralNetwork to solve the MNIST data set.
"""
import sys, os
import numpy

# Sloppily add neural_network to our path so we can import it
sys.path.insert(0, os.path.abspath('../neural_network'))

from neural_networkD import NeuralNet

def train_the_neural_net(neural_net, epochs=1):
    print 'Training the neural network.'
    training_data_file = open('mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = epochs
    for i in range(epochs):
        print 'Training epoch {}/{}.'.format(i+1, epochs)
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99

            neural_net.train(inputs, targets)

    print 'complete.'


def test_the_neural_net(neural_net):
    print 'Testing the neural network.'
    test_data_file = open('mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    scorecard = []
    for i, record in enumerate(test_data_list):
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        outputs = neural_net.query(inputs)

        label = numpy.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    print 'complete.'

    return scorecard


if __name__ == '__main__':

    print 'Starting neural network to recognize handwritten digits.'

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1

    nn = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Train
    train_the_neural_net(nn, epochs=1)

    # Test
    test_results = numpy.asarray(test_the_neural_net(nn))

    # Print results
    print('Neural network is {}% accurate at predicting handwritten digits.'
        .format(test_results.sum() / float(test_results.size) * 100.0))
