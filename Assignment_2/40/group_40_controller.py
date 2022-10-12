import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from controller import Controller
import numpy as np

NEURONS=10

INPUTS=20

OUTPUTS=5

POP_SIZE=10

total_weights = (INPUTS + 1) * NEURONS + (NEURONS + 1) * (OUTPUTS)

class group40Controller(Controller):

    def __init__(self, weights):
        self.bias = np.reshape(weights[0:NEURONS], (NEURONS,))

        self.input = np.reshape(weights[NEURONS:INPUTS * NEURONS + NEURONS], (INPUTS, NEURONS))

        #self.bias = np.reshape(weights[INPUTS * NEURONS: INPUTS * NEURONS + NEURONS], (NEURONS,))
        self.bias2 = np.reshape(weights[INPUTS * NEURONS + NEURONS: INPUTS * NEURONS + NEURONS + OUTPUTS], (OUTPUTS,))
        self.output = np.reshape(weights[INPUTS * NEURONS + NEURONS + OUTPUTS:], (NEURONS, OUTPUTS))


    def control(self, params, cont):
        normal = (params - np.min(params)) / (np.max(params) - np.min(params))

        first = (normal @ self.input) + self.bias

        activate = 1 / (1 + np.exp(-first))

        output = activate @ self.output + self.bias2

        output_activation = 1 / (1 + np.exp(-output))

        return output_activation > 0.5
