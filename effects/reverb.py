

import numpy as np

from .delay import Delay
from .tapped_delay_line import TappedDelayLine
from .comb import Comb
from .allpass import Allpass

M_LN10 = 2.30258509299404568402 # natural log of 10
def dB_to_linear(x):
    return np.exp((x)*M_LN10 * 0.05)

class ReverbConfig():
    def __init__(self):
        self.room_scale =   50  # %
        self.pre_delay  =   0   # ms
        self.wet_gain   =   0   # dB
        self.dry_gain   =   0   # dB
        self.hf_damping =   0   # %
        self.reverberance = 50  # %
        self.stereo_width = 0   # %
        self.er_gain =      0   # %

'''        
             ___________      ___________________      ______          _________
            |           |    |                   |    |      |        |         |
 x[n] ----> | Pre-Delay | -> | Early Reflections | -> | Comb | x 8 -> | Allpass | x 4 -----> y[n]
            |___________|    |___________________|    |______|        |_________|
'''

tap_delays = [190,  949,  993,  1183, 1192, 1315,
                2021, 2140, 2524, 2590, 2625, 2700,
                3119, 3123, 3202, 3268, 3321, 3515]

tap_gains = [.841, .504, .49,  .379, .38,  .346,
                .289, .272, .192, .193, .217,  .181,
                .18,  .181, .176, .142, .167, .134]

comb_lengths = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
allpass_lengths = [556, 441, 341, 225]

class Reverb():
    def __init__(self, config, sample_rate = 44100, num_channels = 1):
        # Audio formats
        self.num_channels = num_channels
        r = sample_rate / 44100

        # Configurations
        delay_length = int(config.pre_delay / 1000 * sample_rate + .5)
        scale = config.room_scale / 100 * .9 + .1
        width = config.stereo_width / 100
        a = -1 / np.log(1 -  .3)
        b = 100 / (np.log(1 - .98) * a + 1)
        feedback = 1 - np.exp((config.reverberance - b) / (a * b))
        hf_damping = config.hf_damping / 100 * .3 + .2

        self.dry_gain = dB_to_linear(config.dry_gain)

        self.wet_gain = dB_to_linear(config.wet_gain)
        self.wet1 = self.wet_gain * (width / 2 + 0.5)
        self.wet2 = self.wet_gain * ((1 - width) / 2)

        self.er_gain = config.er_gain
        self.er1 = self.er_gain * (width / 2 + 0.5)
        self.er2 = self.er_gain * ((1 - width) / 2)

        stereo = num_channels == 2
        
        # Pre-delay buffer
        self.delay_left = Delay(delay_length)
        if stereo : self.delay_right = Delay(delay_length)

        # Early reflections by tapped delay line
        self.er_left = TappedDelayLine(tap_delays, tap_gains) 
        if stereo : self.er_right = TappedDelayLine(tap_delays, tap_gains) 

        # Parallel comb filters
        self.combs_left = []
        self.combs_right = []
        for length in comb_lengths:
            self.combs_left.append(Comb(int(length * r * scale + 0.5), feedback, hf_damping))   
            if stereo : self.combs_right.append(Comb(int(length * r * scale + 0.5), feedback, hf_damping))   

        # Cascaded allpass filters
        self.allpasses_left = []
        self.allpasses_right = []
        for length in allpass_lengths:
            self.allpasses_left.append(Allpass(int(length * r + 0.5), 0.5))
            if stereo : self.allpasses_right.append(Allpass(int(length * r + 0.5), 0.5))

    def process(self, x_l, x_r = None):
        if self.num_channels == 2 and x_r == None:
            return -1
        input_l = self.delay_left.front()
        self.delay_left.push(x_l)
        if self.num_channels == 2:
            input_r = self.delay_right.front()
            self.delay_right.push(x_r)
            input = 0.5 * (input_l + input_r) * 0.015
        else:
            input = input_l * 0.015

        er_l = 0
        er_r = 0
        out_l = 0
        out_r = 0

        er_l = self.er_left.process(input)

        for comb in self.combs_left:
            out_l += comb.process(er_l)

        for allpass in self.allpasses_left:
            out_l = allpass.process(out_l)


        if self.num_channels == 2:
            er_r = self.er_right.process(input)

            for comb in self.combs_right:
                out_r += comb.process(er_r)

            for allpass in self.allpasses_right:
                out_r = allpass.process(out_r)

            y_l = self.er1 * er_l + self.er2 * er_r + self.wet1 * out_l + self.wet2 * out_r + self.dry_gain * x_l
            y_r = self.er2 * er_l + self.er1 * er_r + self.wet2 * out_l + self.wet1 * out_r + self.dry_gain * x_r
            return y_l, y_r

        else :
            y = self.er_gain * er_l + self.wet_gain * out_l + self.dry_gain * x_l
            return y
        

