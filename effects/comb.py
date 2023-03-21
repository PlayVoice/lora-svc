
from .delay import Delay

'''
            _      +------------------> y
           | |     | 
    x ---->|+|-----+--->[z^-M]------+                       
           |_|                      |
            ^                       |
            |                       |
            +------[feedback]<------+
'''
class Comb():
    def __init__(self, delay_length, feedback, damp):
        self.feedback = feedback
        self.damp = damp
        self.delay = Delay(delay_length)
        self.store = 0

    def set_feedback(self, feedback):
        self.feedback = feedback

    def set_damp(self, damp):
        self.damp = damp

    def process(self, x):
        y_delay = self.delay.front()
        self.store = y_delay * (1. - self.damp) + self.store * self.damp
        y = x + self.store * self.feedback
        self.delay.push(y)
        return y
