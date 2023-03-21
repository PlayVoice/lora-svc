import numpy as np
import math
from .delay import Delay
from .lfo import LFO

'''      
                                                              ___
    x -----+----------------------------------dry gain------>|   |
           |                                                 |   |
           +----->[ delay 1 ]------------------gain 1------->|   |
           |          /|\                                    |   |
           |           |                                     | + |-------> y
           |        [ LFO ]                                  |   |
           ：                                                :   :
           ：                                                :   :
           |                                                 |   |
           +--------------->[ delay n ]--------gain n------->|___|
                               /|\                             
                                |                               
                             [ LFO ]                           
                                                              
'''

def validate(sample_rate, delays, mod_freqs, mod_depths, chorus_gains, dry_gain):

    return

class Chorus():
    def __init__(self, sample_rate, delays, mod_freqs, mod_width, chorus_gains, dry_gain=1):
        validate(sample_rate, delays, mod_freqs, mod_width, chorus_gains, dry_gain)
        self.sample_rate = sample_rate
        self.chorus_count = len(chorus_gains)
        self.chorus_gains = chorus_gains
        self.dry_gain = dry_gain

        # Multiple chorus
        max_delay = 0
        self.lfo_array = []
        self.chorus_delays = np.zeros(self.chorus_count)
        for i in range(self.chorus_count):
            self.chorus_delays[i] = math.floor(sample_rate * delays[i])
            width = math.floor(sample_rate * mod_width[i])
            max_delay_i = self.chorus_delays[i] + width + 2
            if max_delay_i > max_delay:
                max_delay = max_delay_i
            self.lfo_array.append(LFO(sample_rate, mod_freqs[i], width))   
        
        self.delay_line = Delay(int(max_delay))  # one delay line used for all paths
        return

    def process(self, x):
        y = x * self.dry_gain
        for i in range(self.chorus_count):
            lfo = self.lfo_array[i]
            tap = self.chorus_delays[i] + lfo.tick()
            d = math.floor(tap)

            # Linear Interpolation 
            frac = tap - d
            candidate1 = self.delay_line.go_back(d)
            candidate2 = self.delay_line.go_back(d + 1)
            interp = frac * candidate2 + (1 - frac) * candidate1
            y += interp * self.chorus_gains[i]
        self.delay_line.push(x)
        return y



