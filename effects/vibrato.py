import math
from .delay import Delay
from .lfo import LFO
'''                                                      
    x ----------->[ delay 1 ]---------> y
                      /|\                                  
                       |                                   
                    [ LFO ]                                                                                
'''
class Vibrato():
    def __init__(self, sample_rate, delay, mod_width, mod_freq):
        self.sample_rate = sample_rate
        self.avg_delay = math.floor(sample_rate * delay)
        width = math.floor(sample_rate * mod_width)
        if self.avg_delay < width:
            self.avg_delay = width
        max_delay = self.avg_delay + 2 * width + 3
        self.delay_line = Delay(max_delay)
        self.lfo = LFO(sample_rate, mod_freq, width)
        return

    def process(self, x):
        tap = self.avg_delay + self.lfo.tick()
        i = math.floor(tap)

        # Linear Interpolation 
        frac = tap - i
        candidate1 = self.delay_line.go_back(i)
        candidate2 = self.delay_line.go_back(i + 1)
        interp = frac * candidate2 + (1 - frac) * candidate1

        self.delay_line.push(x)
        return interp


