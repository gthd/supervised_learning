import numpy as np
import pickle
from collections import namedtuple

Buffer = namedtuple('Buffer',
                        ('label', 'grasp_pos_x', 'grasp_pos_y', 'grasp_orientation', 'image'))

class ReplayMemory(object):

    def __init__(self, capacity): #capacity of the one file
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.buffer_num = 46
        self.pointer = 0
        self.low = 0
        self.flag = False
        self.indices = np.arange(self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Buffer(*args)
        self.position = (self.position + 1) % self.capacity
        self.pointer += 1
        if self.pointer > self.capacity:
            self.pointer = 0
            self.buffer_num += 1

    def len(self):
        return len(self.memory)

    def empty(self):
        self.memory = []
        self.position = 0

    def store_at_disk(self):
        data_file = open('/homes/gt4118/Desktop/supervised_learning/Datasets/my_dataset'+str(self.buffer_num)+'.pkl', 'ab')
        pickle.dump(self.memory[0], data_file, -1)
        data_file.close()
