import math
import random
import sys, os
from typing import List

from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QListWidget, \
    QListWidgetItem, QMessageBox, QMainWindow, QComboBox

import numpy as np
from matplotlib import pyplot

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
        self.herbsOnTick = 5
        self.startAgentCount = 10
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.units: List[Agent] = []

    def tick(self):
        for agent in self.units:
            agent.doSomething()
            agent.starve()
            if agent.isReadyToBreed():
                self.units.append(agent.breed())

        self.units=list(filter(lambda unit: unit.isDead()==False, self.units))

        for _ in range(self.herbsOnTick):
            x, y = random.randint(self.min_x,self.max_x), random.randint(self.min_y, self.min_x)
            herb = self.createAgent(TYPES[2],x,y)
            self.units.append(herb)

        self.units.de
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

    def createAgent(self, agent_type, pos_x, pos_y):
        new_agent = None

        if agent_type == "Carnivore":
            new_agent = Carnivore(pos_x, pos_y, self)
        elif agent_type == "Herbivore":
            new_agent = Herbivore(pos_x, pos_y, self)
        elif agent_type == "Herb":
            new_agent = Herb(pos_x, pos_y, self)

        return new_agent

    def initAgents(self):
        herbsCount= 10
        carnivoreCount= 1
        herbivoreCount= 4

        for _ in range(carnivoreCount):
            x, y = random.randint(self.min_x,self.max_x), random.randint(self.min_y, self.min_x)
            new_agent = self.createAgent(TYPES[0],x,y)
            self.units.append(new_agent)

        for _ in range(herbivoreCount):
            x, y = random.randint(self.min_x,self.max_x), random.randint(self.min_y, self.min_x)
            self.createAgent(TYPES[1],x,y)
            self.units.append(new_agent)

        for _ in range(herbsCount):
            x, y = random.randint(self.min_x,self.max_x), random.randint(self.min_y, self.min_x)
            self.createAgent(TYPES[2],x,y)
            self.units.append(new_agent)


class Food:
    def __init__(self, type, points):
        self.type = type
        self.points = points


class Agent:

    def __init__(self, pos_x, pos_y, world: World, agent_type, movable: bool, breedable: bool, ration: list, health=1):

        self.health_breed = 10
        self.starvingDamage = 10
        self.max_health = 100
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

        self.visionZone = AgentVisionArea()
        self.movable = movable
        self.breedable = breedable
        self.ration = ration

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

    def addHp(self, hp):
        if self.health + hp > self.max_health:
            self.health = self.max_health
        else:
            self.health += hp

    def starve(self):
        if len(self.ration) != 0:
            self.health -= self.starvingDamage

    def kill(self):
        self.health = 0

    def isDead(self):
        return self.health <= 0

    def isReadyToBreed(self):
        return self.breedable and self.health >= self.health_breed

    def breed(self):
        newAgent = self.world.createAgent(self.agent_type, self.pos_x, self.pos_y)
        newAgent.health = self.health / 2
        for neuron in newAgent.neurons:
            neuron.mutate()

        return newAgent

    def eat(self):
        nearAgents = self.visionZone.getNearAgents()
        for agent in nearAgents:
            for food in self.ration:
                if food.type == agent.agent_type:
                    self.addHp(food.points)
                    agent.kill()
                    return

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


class Carnivore(Agent):
    def __init__(self, pos_x, pos_y, world):
        food = Food("Herbivore", 10)
        super().__init__(pos_x, pos_y, world, "Carnivore", True, True, [food])


class Herbivore(Agent):
    def __init__(self, pos_x, pos_y, world):
        food = Food("Herb", 10)
        super().__init__(pos_x, pos_y, world, "Herbivore", True, True, [food])


class Herb(Agent):
    def __init__(self, pos_x, pos_y, world):
        super().__init__(pos_x, pos_y, world, "Herb", False, False, [])


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

    def calcDistance(self, source_cord, target_cord, worldSize):
        dCord = target_cord - source_cord
        if abs(dCord) > worldSize / 2:
            dcordAbs = worldSize - abs(dCord)
            return dcordAbs if dCord < 0 else -dcordAbs
        else:
            return dCord

    def distanceTo(self, source: Agent, target: Agent):
        world = source.world
        dx = self.calcDistance(source.pos_x, target.pos_x, world.max_x - world.min_x + 1)
        dy = self.calcDistance(source.pos_y, target.pos_y, world.max_y - world.min_y + 1)
        return dx, dy


class AgentVisionArea:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.positioning = Positioning()

    def isOnFront(self, target: Agent) -> bool:
        """Проверка нахождения target перед self.agent"""
        dx, dy = self.positioning.distanceTo(self.agent, target)
        return -2 <= dx <= 2 and dy == 2

    def isOnLeft(self, target: Agent) -> bool:
        """Проверка нахождения target слева от self.agent"""
        dx, dy = self.positioning.distanceTo(self.agent, target)
        return 0 <= dy <= 1 and dx == -2

    def isOnRight(self, target: Agent) -> bool:
        """Проверка нахождения target справа от self.agent"""
        dx, dy = self.positioning.distanceTo(self.agent, target)
        return 0 <= dy <= 1 and dx == 2

    def isNear(self, target: Agent) -> bool:
        """Проверка нахождения target в близости self.agent"""
        dx, dy = self.positioning.distanceTo(self.agent, target)
        return 0 <= dy <= 1 and -1 <= dx <= 1

    def whereIs(self, target: Agent):
        if self.isOnFront(target):
            return target.agent_type + "Front"
        elif self.isOnLeft(target):
            return target.agent_type + "Left"
        elif self.isOnRight(target):
            return target.agent_type + "Right"
        elif self.isNear(target):
            return target.agent_type + "Near"
        else:
            return None

    def getNearAgents(self):
        agentsNear = []
        agents = self.agent.world.units

        for agent in agents:
            if agent != self.agent and self.isNear(agent):
                agentsNear.append(agent)

        return agentsNear

    def getInputData(self):
        inputData = []
        for unit in self.agent.world.units:
            if unit != self.agent:
                inputType = self.whereIs(unit)
                if inputType is not None:
                    for signal in surroundings.keys():
                        if surroundings[signal] == inputType:
                            inputData[signal] += 1
                            break

        return inputData


if __name__ == '__main__':
    print("start")
