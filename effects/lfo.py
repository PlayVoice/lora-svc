import math
'''      
    Low Frequency Oscillator
'''
class LFO():
    def __init__(self, sample_rate, frequency, width, waveform='sine', offset=0, bias=0):
        self.waveform = waveform
        self.width = width
        self.delta = frequency / sample_rate
        self.phase = offset
        self.bias = bias
        return

    def process(self, n):
        return self.width * math.sin(2 * math.pi * self.delta * n) + self.bias

    def tick(self, i=1):
        ret = self.width * math.sin(2 * math.pi * self.phase) + self.bias

        self.phase += i * self.delta
        if( self.phase > 1.0):
            self.phase -= 1.0
        return ret
