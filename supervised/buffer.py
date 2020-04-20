import pickle
from collections import namedtuple
import random
import numpy as np

Buffer = namedtuple('Buffer',
                    ('label', 'grasp_pos_x', 'grasp_pos_y', 'grasp_orientation', 'image'))

class ReplayMemory(object):

    def __init__(self, capacity): #capacity of the one file
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.buffer_num = 0
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
            self.memory = []

    def len(self):
        return len(self.memory)

    def store_at_disk(self):
        data_file = open('/home/george/Desktop/Github/supervised_learning/Datasets/my_dataset'+ \
            str(self.buffer_num)+'.pkl', 'ab')
        pickle.dump(self.memory[self.position-1], data_file, -1)
        data_file.close()
