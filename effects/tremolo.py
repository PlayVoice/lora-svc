from .lfo import LFO
'''                                                      
    x ----------->[x]---------> y
                  /|\                                  
                   |            
                  gain
                  /|\                                  
                   |                                   
                [ LFO ]                                                                                
'''
class Tremolo():
    def __init__(self, mod_freq, mod_depth = 0.5, sample_rate=44100):
        self.lfo = LFO(sample_rate, mod_freq, mod_depth * 0.5)
        self.mod_depth = mod_depth
        return

    def process(self, x):
        gain = 1 - 0.5 * self.mod_depth - self.lfo.tick()
        return x * gain


