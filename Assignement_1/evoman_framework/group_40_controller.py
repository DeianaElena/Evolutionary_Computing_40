import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
import numpy as np

NEURONS=10

INPUTS=20

OUTPUTS=5

POP_SIZE=10

total_weights = INPUTS * NEURONS + NEURONS + NEURONS * OUTPUTS

class group40Controller(Controller):

    def __init__(self, weights):
        self.input = np.reshape(weights[0:INPUTS * NEURONS], (INPUTS, NEURONS))
        self.bias = np.reshape(weights[INPUTS * NEURONS: INPUTS * NEURONS + NEURONS], (NEURONS,))
        self.output = np.reshape(weights[INPUTS * NEURONS + NEURONS:], (NEURONS, OUTPUTS))

    def control(self, params, cont):
        first = (params @ self.input) + self.bias

        activate = 1 / (1 + np.exp(-first))

        output = activate @ self.output

        output_activation = 1 / (1 + np.exp(-output))

        return output > 0.5
