
from .delay import Delay

'''
                  ————>[-feedback]———> 
            _    |                    | 
     x ———>|+|———+——>[z^-M]—————————>|+|——> y
            |                    |
             <—————[feedback]<———
'''
class Allpass():
    def __init__(self, delay_length, feedback):
        self.feedback = feedback
        self.delay = Delay(delay_length)

    def set_feedback(self, feedback):
        self.feedback = feedback

    def process(self, x):
        v_delay = self.delay.front()
        v = self.feedback * v_delay + x
        y = v_delay - self.feedback * v
        self.delay.push(v)
        return y
        