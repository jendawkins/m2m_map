from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    def __init__(self, x, y, i):
        self.x = x
        self.y = y
        self.i = i

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        pyplot.gca().text(self.x-0.1, self.y-0.1, str(self.i))


class Layer():
    def __init__(self, network, number_of_neurons, weights, label = None, weight_label = None, node_labels = None):
        self.net = network
        self.node_labels = node_labels
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        self.label = label
        self.weight_label = weight_label

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            if self.node_labels:
                neuron = Neuron(x, self.y, self.node_labels[iteration])
            else:
                neuron = Neuron(x, self.y, iteration)
            neurons.append(neuron)
            x += self.net.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.net.horizontal_distance_between_neurons * (self.net.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.net.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.net.neuron_radius * sin(angle)
        y_adjustment = self.net.neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        pyplot.gca().text(self.x-2*self.net.horizontal_distance_between_neurons, self.y, self.label)
        if self.previous_layer:
            pyplot.gca().text(self.x-2*self.net.horizontal_distance_between_neurons, self.y - self.net.vertical_distance_between_layers/2, self.weight_label)
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw(self.net.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                    previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                    weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)


class NeuralNetwork():
    def __init__(self, vertical_distance_between_layers = 6,
                 horizontal_distance_between_neurons = 2, neuron_radius = 0.5, number_of_neurons_in_widest_layer = 10):
        self.layers = []
        self.vertical_distance_between_layers = vertical_distance_between_layers
        self.horizontal_distance_between_neurons = horizontal_distance_between_neurons
        self.neuron_radius = neuron_radius
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer

    def add_layer(self, number_of_neurons, weights=None, label = None, weight_label = None, node_labels = None):
        layer = Layer(self, number_of_neurons, weights, label = label, weight_label = weight_label, node_labels = node_labels)
        self.layers.append(layer)

    def draw(self, fname):
        for layer in self.layers:
            layer.draw()
        # pyplot.gca().set_xlim([-3*self.horizontal_distance_between_neurons,
        #                    self.horizontal_distance_between_neurons*self.number_of_neurons_in_widest_layer + 3*self.horizontal_distance_between_neurons])
        pyplot.axis('scaled')
        # pyplot.gca().set(frame_on=False)
        pyplot.axis('off')
        pyplot.savefig(fname)
        pyplot.close()


if __name__ == "__main__":
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
    weights1 = np.array([\
                         [0,0,0,0,0,0,0,0,1,1],\
                         [0,0,0,0,1,1,1,1,0,0],\
                         [0,0,1,1,0,0,1,1,0,0],\
                         [0,1,0,1,0,1,0,1,0,1]])
    network.add_layer(10, weights1, label = 'test layer')
    network.add_layer(4, label = 'test', weight_label = 'test weights')
    network.draw('test.pdf')