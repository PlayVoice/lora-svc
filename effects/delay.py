import numpy as np
'''
     push ------>[z^-M]-----> front
'''
class Delay():
    def __init__(self, delay_length):
        self.length = delay_length
        self.buffer = np.zeros(delay_length)
        self.pos = 0

    def front(self):
        return self.buffer[self.pos]
    
    def push(self, x):
        self.buffer[self.pos] = x
        self.pos = self.pos + 1
        if self.pos + 1 >= self.length :
            self.pos = self.pos - self.length 
    
    def go_back(self, idx):
        target = self.pos - idx
        if target < 0 :
            target = target + self.length 
        return self.buffer[target]
