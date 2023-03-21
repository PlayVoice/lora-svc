import math
'''      
    Fade-in and  Fade-out
'''
def FadeGain(index, range, type='linear'):
    rate = index / range
    if rate > 1.0:
        rate = 1.0
    elif rate < 0.0:
        rate = 0.0
    if (type == 'linear'):
        gain  = rate
    elif (type == 'sin'):
        gain  = math.sin(rate * math.pi * 0.5)
    else:
        gain  = math.cos(rate * math.pi * 0.5)
    return gain

class Fade():
    def __init__(self, sample_rate, in_length, out_start, out_length):
        self.in_end = math.floor(in_length * sample_rate)
        self.out_start = math.floor(out_start * sample_rate)
        self.out_end = math.floor((out_start  + out_length) * sample_rate)
        self.count = 0
        return

    def gain(self):
        if (self.count < self.in_end):
            return FadeGain(self.count, self.in_end)
        elif (self.count < self.out_start):
            return 1.0
        elif (self.count < self.out_end):
            return FadeGain(self.out_end - self.count, self.out_end - self.out_start)
        else:
            return 0.0

    def process(self, x):
        gain = self.gain()
        self.count += 1
        return x * gain

class FadeIn():
    def __init__(self, sample_rate, in_length):
        self.in_length = math.ceil(in_length * sample_rate)
        self.count = 0
        return

    def gain(self):
        if (self.count < self.in_length):
            return FadeGain(self.count, self.in_length)
        else:
            return 1.0

    def process(self, x):
        gain = self.gain()
        self.count += 1
        return x * gain

class FadeOut():
    def __init__(self, sample_rate, out_length):
        self.out_length = math.ceil(out_length * sample_rate)
        self.count = 0
        return

    def gain(self):
        if (self.count < self.out_length):
            return FadeGain(self.out_length - self.count, self.out_length)
        else:
            return 0.0

    def process(self, x):
        gain = self.gain()
        self.count += 1
        return x * gain

