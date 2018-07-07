import random
from train import train_and_score


class Network:

    def __init__(self, nn_param_choice):
        self.nn_param_choices = nn_param_choice
        self.network = {}

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """
        :param network dict: dictionary with network parameters
        :return:
        """
        self.network = network

    def train(self, x_train, y_train, x_test, y_test):
        self.accuracy = train_and_score(self.network, x_train, y_train, x_test, y_test)
