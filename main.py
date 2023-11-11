import math
import sys, os

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QListWidget, \
    QListWidgetItem, QMessageBox, QMainWindow, QComboBox

import numpy as np
from matplotlib import pyplot


class Neuron:

    def __init__(self, num_inputs):
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.output = 0

    def calc_output(self, inputs):
        self.output = np.sum(inputs * self.weights)
        return self.output

    def calc_error(self, expexted_output):
        return expexted_output - self.output

    def update_weights(self, learning_rate, error, inputs):
        self.weights += learning_rate * error * inputs


class World:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.units = []

    def changePos(self, x, y):
        new_x = x
        new_y = y

        if x < self.min_x:
            new_x = self.max_x
        elif x > self.max_x:
            new_x = self.min_x

        if y < self.min_y:
            new_y = self.max_y
        elif y > self.max_y:
            new_y = self.min_y

        return new_x, new_y


SIDES = {0: "Front", 1: "Right", 2: "Left", 3: "Near"}
TYPES = {0: "Carnivore", 1: "Herbivore", 2: "Herb"}
#   surroundings = {
#   0: 'CarnivoreUp', 1: 'CarnivoreRight', 2: 'CarnivoreDown', 3: 'CarnivoreLeft',
#   4: 'HerbivoreUp', 5: 'HerbivoreRight', 6: 'HerbivoreDown', 7: 'HerbivoreLeft',
#   8: 'HerbUp', 9: 'HerbRight', 10: 'HerbDown', 11: 'HerbLeft'
#   }
surroundings = {}
loopcount = 0
for t in TYPES.values():
    for side in SIDES.values():
        surroundings[loopcount] = t + side
        loopcount += 1

print(surroundings)


class Agent:

    def __init__(self, pos_x, pos_y, world: World, agent_type, health=1):
        self.neuron_count = 4
        self.neuron_size = 12
        self.world = world

        # lookingAt:
        #   0: Up,
        #   1: Right,
        #   2: Down,
        #   3: Left
        self.lookingAt = 0

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.health = health
        self.agent_type = TYPES[agent_type]
        self.neurons = [Neuron(self.neuron_size) for _ in range(self.neuron_count)]

    def turn_left(self):
        if self.lookingAt > 0:
            self.lookingAt -= 1
        else:
            self.lookingAt = 3

    def turn_right(self):
        if self.lookingAt < 3:
            self.lookingAt += 1
        else:
            self.lookingAt = 0

    def move(self):
        positioner = Positioning()
        pos = positioner.moveAgent(agent=self)

    def eat(self):
        pass

    def doSomething(self, inputs):
        if len(inputs) != self.neuron_size:
            print("Error, incorrect input size")

        maxOutput = -9999999.0
        action = 0

        # noinspection PyTypeChecker
        for i in len(self.neurons):
            output = self.neurons[i].calc_output(inputs)
            if output > maxOutput:
                maxOutput = output
                action = i

        # actions:
        # 0: "TurnLeft",
        # 1: "TurnRight",
        # 2: "Move",
        # 3: "Eat"
        if action == 0:
            self.turn_left()
        elif action == 1:
            self.turn_right()
        elif action == 2:
            self.move()
        elif action == 3:
            self.eat()


class Positioning:
    def moveAgent(self, agent: Agent):
        # lookingAt:
        #   0: Up,
        #   1: Right,
        #   2: Down,
        #   3: Left
        x = agent.pos_x
        y = agent.pos_y

        if agent.lookingAt == 0:
            y += 1
        elif agent.lookingAt == 1:
            x += 1
        elif agent.lookingAt == 2:
            y -= 1
        elif agent.lookingAt == 3:
            x -= 1

        return agent.world.changePos(x, y)

    def distanceTo(self, source: Agent, target: Agent):
        world = source.world

    def ToroidalDistance(self, source_cord, target_cord, worldSize):
        dCord = source_cord - target_cord
        if abs(dCord) > worldSize / 2:
            dcordAbs = worldSize - (abs(dCord) + 1)
            return dcordAbs if dCord < 0 else -dcordAbs
        else:
            return dCord

if __name__ == '__main__':
    print("start")
