import random
from pyglet import app, image, clock, text
from pyglet.window import Window
import numpy as np
import pickle

window = Window(1000, 1000)
cell_size = 20
mutation_rate = 0.6

iNodes = 24
hNodes = 16
hLayers = 2
oNodes = 4

replay_best = True



def draw_square(x, y, size, colour=(255, 255, 255, 0)):
    img = image.create(size, size, image.SolidColorImagePattern(colour))
    img.blit(x, y)


def activatefunction(x):
    return max(0, x)  # ReLu


def mutation(x):
    if random.random() < mutation_rate:
        x += np.random.normal(0, 0.7)
    return x


def activate(a):
    vfunc = np.vectorize(activatefunction)
    return vfunc(a)


def mutate(a):
    vfunc = np.vectorize(mutation)
    return vfunc(a)


def crossover(a, b):
    child = b.copy()
    rows = len(a)
    cols = len(a[0])
    rr = random.randint(0, rows)
    rc = random.randint(0, cols)
    for i in range(rows):
        for j in range(cols):
            if i < rr or (i == rr and j <= rc):
                child[i][j] = a[i][j]
    return child


def wallcollide(x, y):
    if x < 0 or x >= window.width:
        return True
    if y < 0 or y >= window.height:
        return True
    return False


class NeuralNet:
    def __init__(self, inp, hidden, hiddenlayers, output):
        self.input = inp
        self.hidden = hidden
        self.hiddenlayers = hiddenlayers
        self.output = output
        self.weights = [0] * (self.hiddenlayers + 1)
        #self.weights = np.zeros(self.hiddenlayers + 1, dtype=object)
        x = np.vectorize(lambda a: 2 * a - 1)
        self.weights[0] = x(np.random.rand(self.input + 1, self.hidden))
        for i in range(1, self.hiddenlayers):
            self.weights[i] = x(np.random.rand(self.hidden, self.hidden + 1))
        self.weights[-1] = x(np.random.rand(self.hidden + 1, self.output))

    def mutate(self):
        self.weights = list(map(mutate, self.weights))

    def outp(self, inp):
        for i in range(self.hiddenlayers + 1):
            inp = inp + [1]
            inp = activate(np.dot(inp, self.weights[i]))
        return inp

    def crossover(self, partner):
        child = NeuralNet(self.input, self.hidden, self.hiddenlayers, self.output)
        for i in range(self.hiddenlayers + 1):
            child.weights[i] = crossover(self.weights[i], partner.weights[i])
        return child

    def load(self, a):
        self.weights = a

    def clone(self):
        a = NeuralNet(self.input, self.hidden, self.hiddenlayers, self.output)
        a.weights = self.weights.copy()
        return a

    def save_weights(self):
        #np.savetxt("model.txt", self.weights)
        pickle.dump(self.weights, open('model.pkl', 'wb'))


class Population:
    def __init__(self, size):
        self.size = size
        self.snakes = [Snake() for i in range(size)]
        self.best_snake = self.snakes[0].clone()
        self.best_score = 0
        self.gen = 0
        self.samebest = 0
        self.best_fitness = 0
        self.fitness_sum = 0
        self.best_snake.replay = True

    def done(self):
        for i in range(self.size):
            if not self.snakes[i].dead:
                return False
        if not self.best_snake.dead:
            return False
        return True

    def update(self):
        if not self.best_snake.dead:
            self.best_snake.look()
            self.best_snake.think()
            self.best_snake.move()
        for i in range(self.size):
            if not self.snakes[i].dead:
                self.snakes[i].look()
                self.snakes[i].think()
                self.snakes[i].move()

    def show(self):
        if replay_best:
            self.best_snake.show()
            label = text.Label('Gen: ' + str(self.gen),
                               font_name='Roboto',
                               font_size=20,
                               x=10, y=10)
            label.draw()
        else:
            for i in range(self.size):
                self.snakes[i].show()

    def set_best_snake(self):
        mx = 0
        max_index = 0
        for i in range(self.size):
            if self.snakes[i].fitness > mx:
                mx = self.snakes[i].fitness
                max_index = i
        if mx > self.best_fitness:
            self.best_fitness = mx
            self.best_snake = self.snakes[max_index].cloneforreplay()
            self.best_score = self.snakes[max_index].score
            #self.samebest = 0
        else:
            self.best_snake = self.best_snake.cloneforreplay()
            #self.samebest += 1
            #if self.samebest >= 2:
                #global mutation_rate
                #mutation_rate *= 2
                #self.samebest = 0

    def select_parent(self):
        rand = random.randint(0, self.fitness_sum)
        sm = 0
        for i in range(self.size):
            sm += self.snakes[i].fitness
            if sm > rand:
                return self.snakes[i]
        return self.snakes[0]

    def natural_selection(self):
        new_snakes = [Snake() for i in range(self.size)]
        self.set_best_snake()
        self.calculate_fitness_sum()
        new_snakes[0] = self.best_snake.clone()
        for i in range(1, self.size):
            child = self.select_parent().crossover(self.select_parent())
            child.mutate()
            new_snakes[i] = child
        self.snakes = new_snakes.copy()
        self.gen += 1

    def mutate(self):
        for i in range(1, self.size):
            self.snakes[i].mutate()

    def calculate_fitness(self):
        for i in range(0, self.size):
            self.snakes[i].calculateFitness()

    def calculate_fitness_sum(self):
        self.fitness_sum = 0
        for i in range(0, self.size):
            self.fitness_sum += self.snakes[i].fitness


class Food:
    def __init__(self):
        self.x = random.randint(0, (window.width // cell_size) - 1) * cell_size
        self.y = random.randint(0, (window.height // cell_size) - 1) * cell_size

    def show(self):
        x = random.randint(0, 127)
        y = random.randint(0, 127)
        z = random.randint(0, 127)
        draw_square(self.x, self.y, cell_size, colour=(255, 0, 0, 0))

    def clone(self):
        clone = Food()
        clone.x = self.x
        clone.y = self.y
        return clone


class Snake:
    def __init__(self, foods=None):
        if foods is None:
            self.score = 1
            self.time_left = 400
            self.lifetime = 0
            self.dx, self.dy = 0, 0
            self.fitness = 0.
            self.i = 0
            self.dead = False
            self.replay = False

            self.body = [[window.width // cell_size // 2 * cell_size, window.height // cell_size // 2 * cell_size]]
            self.food = Food()
            self.foodList = [self.food.clone()]

            self.brain = NeuralNet(iNodes, hNodes, hLayers, oNodes)
            self.vision = [0] * 24
            self.decision = [0] * 4
        else:
            self.score = 1
            self.time_left = 200
            self.lifetime = 0
            self.dx, self.dy = 0, 0
            self.fitness = 0.
            self.i = 0
            self.dead = False
            self.replay = True

            self.body = []
            self.body.append([window.width // cell_size // 2 * cell_size, window.height // cell_size // 2 * cell_size])
            self.foodList = foods.copy()
            self.food = self.foodList[self.i]

            self.vision = [0] * 24
            self.decision = [0] * 4


    def foodcollide(self, a, b):
        if a == self.food.x and b == self.food.y:
            return True
        return False

    def bodycollide(self, x, y):
        for i in range(1, len(self.body)):
            if x == self.body[i][0] and y == self.body[i][1]:
                return True
        return False

    def show(self):
        window.clear()
        self.food.show()
        x = 127
        for coords in self.body:
            draw_square(coords[0], coords[1], cell_size, colour=(x, x, x, 0))

    def move(self):
        if not self.dead:
            self.lifetime += 1
            self.time_left -= 1
            if self.foodcollide(self.body[0][0], self.body[0][1]):
                self.eat()
            self.shiftbody()
            if wallcollide(self.body[0][0], self.body[0][1]):
                self.dead = True
            elif self.bodycollide(self.body[0][0], self.body[0][1]):
                self.dead = True
            elif self.time_left <= 0:
                self.dead = True

    def eat(self):
        self.score += 1
        self.time_left = min(self.time_left + 200, 600)
        self.body.append([self.body[-1][0], self.body[-1][1]])
        if not self.replay:
            self.food = Food()
            while self.bodycollide(self.food.x, self.food.y):
                self.food = Food()
            self.foodList.append(self.food)
        else:
            self.i += 1
            self.food = self.foodList[self.i]


    def shiftbody(self):
        tempx, tempy = self.body[0][0], self.body[0][1]
        tempx += self.dx
        tempy += self.dy
        if self.body:
            self.body.insert(0, [tempx, tempy])
            self.body.pop()

    def clone(self):
        clone = Snake()
        clone.brain = self.brain.clone()
        return clone

    def cloneforreplay(self):
        clone = Snake(self.foodList)
        clone.brain = self.brain.clone()
        return clone

    def crossover(self, parent):
        child = Snake()
        child.brain = self.brain.crossover(parent.brain)
        return child

    def mutate(self):
        self.brain.mutate()

    def calculateFitness(self):
        self.fitness = self.lifetime * (2 ** self.score)
        #if self.score < 10:
        #    self.fitness = self.lifetime ** 2 * (2 ** self.score)
        #else:
        #    self.fitness = self.lifetime ** 2 * (2 ** 10) * (self.score - 9)

    def look(self):
        directions = [[0, cell_size], [cell_size, cell_size], [cell_size, 0], [cell_size, -cell_size],
                      [0, -cell_size], [-cell_size, -cell_size], [-cell_size, 0], [-cell_size, cell_size]]
        for i in range(len(directions)):
            a, b = self.body[0][0], self.body[0][1]
            dist = 0
            bodyfound = 0
            foodfound = 0
            while not wallcollide(a, b):
                a, b = a + directions[i][0], b + directions[i][1]
                dist += 1
                if self.bodycollide(a, b):
                    bodyfound = dist
                if self.foodcollide(a, b):
                    foodfound = dist
            self.vision[3 * i] = foodfound
            self.vision[3 * i + 1] = bodyfound
            self.vision[3 * i + 2] = dist

    def think(self):
        decision = list(self.brain.outp(self.vision))
        direction = decision.index(max(decision))
        if direction == 0:
            if self.dy == 0:
                self.dx = 0
                self.dy = cell_size
        if direction == 1:
            if self.dy == 0:
                self.dx = 0
                self.dy = -cell_size
        if direction == 2:
            if self.dx == 0:
                self.dy = 0
                self.dx = cell_size
        if direction == 3:
            if self.dx == 0:
                self.dy = 0
                self.dx = -cell_size


def update_show(dt):
    window.clear()
    snake.show()
    snake.look()
    snake.think()
    snake.move()


def update_train(dt):
    if pop.done():
        pop.calculate_fitness()
        pop.natural_selection()
        pop.best_snake.brain.save_weights()
    else:
        pop.update()
        pop.show()


train = True

if train:
    pop = Population(2000)
    clock.schedule_interval(update_train, 1 / 50)
else:
    snake = Snake()
    #arr = np.loadtxt("model.txt")
    #snake.brain.weights = arr.reshape(arr.shape[0], arr.shape[1] // arr.shape[2], arr.shape[2])
    snake.brain.weights = pickle.load(open('model.pkl', 'rb'))
    clock.schedule_interval(update_show, 1 / 100)
app.run()






