import math
from .delay import Delay
from .lfo import LFO
'''      
                                                 ___
    x -----+--------------------- -------------->|   |
           |                                    |   |
           |                                    | + |-------> y
           |                                    |   |
           +----->[ delay ]-------gain--------->|   |
                      /|\                       |___|
                       |                          
                    [ LFO ]                                                                                
'''
class Flanger():
    def __init__(self, sample_rate, delay, mod_width, mod_freq, gain=1):
        self.sample_rate = sample_rate
        self.avg_delay = math.floor(sample_rate * delay)
        width = math.floor(sample_rate * mod_width)
        max_delay = self.avg_delay + width + 2
        self.delay_line = Delay(max_delay)
        self.lfo = LFO(sample_rate, mod_freq, width)
        self.gain = gain
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
        return interp * self.gain + x



