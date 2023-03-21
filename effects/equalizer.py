

import numpy as np
from .biquad import Biquad
 
'''
                ___________           
               |           |          
     x[n] ---->| Biquad 1  | X 10 ----> y[n]
               |___________|              
'''
center_freqs = [31.5, 63, 125, 250, 500, 
                1000, 2000, 4000, 8000, 16000]

band_widths = [22, 44, 88, 177, 355, 710, 
               1420, 2840, 5680, 11360]

class Equalizer():
    def __init__(self, gains, sample_rate = 44100):
        # 
        if sample_rate <= 8000:
            self.num_bands = 7
        elif sample_rate <= 16000:
            self.num_bands = 8
        elif sample_rate <= 32000:
            self.num_bands = 9
        else:
            self.num_bands = 10

        # Parallel biquad filters
        self.filters = []

        # A low shelf filter for the lowest band
        self.filters.append(Biquad(sample_rate, 'LowShelf', center_freqs[0],
                                    band_widths[0], gains[0]))  

        # Peaking filters for the middle bands
        for i in range(1, self.num_bands - 1):
            self.filters.append(Biquad(sample_rate, 'Peaking', center_freqs[i],
                                       band_widths[i], gains[i]))   

        # A high shelf filter for the highest band
        self.filters.append(Biquad(sample_rate, 'HighShelf',
                                    center_freqs[self.num_bands - 1],
                                    band_widths[self.num_bands - 1],
                                    gains[self.num_bands - 1])
                            )  

    def process(self, x):
        out = x
        for filter in self.filters:
            out += filter.process(out)
        return out #/ self.num_bands
        
    def dump(self):
        for filter in self.filters:
            filter.dump()
        