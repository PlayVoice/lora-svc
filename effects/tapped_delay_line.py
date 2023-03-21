import numpy as np
'''      
    x ——>[z^-M0]——+—>[z^-(M1-M0)]——+——>[z^-(M2-M1)]——+———— ... [z^-(Mn-Mn-1)]——+
                  |                |                 |                         |
                  g0               g1                g2                        gn
                  |                |                 |                         |    
                  v                v                 v                         v
                  +-------------->[+]-------------->[+]--- ... -------------->[+]---> y
'''
class TappedDelayLine():
    def __init__(self, tap_delays, tap_gains):
        self.delay_length = max(tap_delays)
        self.buffer = np.zeros(self.delay_length)
        self.tap_delays = tap_delays
        self.tap_gains = tap_gains
        self.pos = 0

    def process(self, x):
        y = 0
        self.buffer[self.pos] = x
        for i, delay in enumerate(self.tap_delays):
            gain = self.tap_gains[i]
            delay_idx = self.pos - delay
            if delay_idx < 0:
                delay_idx += self.delay_length
            y += gain * self.buffer[delay_idx]
        
        self.pos = self.pos + 1
        if self.pos >= self.delay_length:
            self.pos = self.pos - self.delay_length
        return y
