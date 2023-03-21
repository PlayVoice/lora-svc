import numpy as np

class Biquad():
    def __init__(
        self, 
        sample_rate, 
        filter_type=None,
        fc=1000,
        bandwidth=1.0,
        gain_db=1.0
    ):
        if sample_rate < 0.0:
            raise ValueError("sample_rate cannot be given a negative value")
        self.sample_rate = sample_rate
    
        self.b = np.zeros(3)
        self.a = np.zeros(3)
        self.a [0] = 1.0
        self.b[0] = 1.0

        self.y = None
        self.x_buf = np.zeros(2)
        self.y_buf = np.zeros(2)
            
        self.filter_type = filter_type

        if fc < 0.0 or fc >= self.sample_rate / 2.0:
            raise ValueError(f"illegal value: fc={fc}")
        self._fc = fc

        self._gain_db = gain_db
        A = 10.0 ** (gain_db/40.0)
        A_add_1 = A + 1.0
        A_sub_1 = A - 1.0
        sqrt_A  = np.sqrt(A)

        w0 = 2.0 * np.pi * self._fc / self.sample_rate
        cos_w0 = np.cos(w0) 
        sin_w0 = np.sin(w0)
        alpha = 0.5 * sin_w0 * fc / bandwidth

        if filter_type == "LowPass":
            self.b[0] = (1.0 - cos_w0) * 0.5
            self.b[1] = (1.0 - cos_w0)
            self.b[2] = (1.0 - cos_w0) * 0.5
            self.a[0] =  1.0 + alpha
            self.a[1] = -2.0 * cos_w0
            self.a[2] =  1.0 - alpha

        elif filter_type == "HighPass":
            self.b[0] =  (1.0 + cos_w0) * 0.5
            self.b[1] = -(1.0 + cos_w0)
            self.b[2] =  (1.0 + cos_w0) * 0.5
            self.a[0] =   1.0 + alpha
            self.a[1] =  -2.0 * cos_w0
            self.a[2] =   1.0 - alpha

        elif filter_type == "BandPass":
            self.b[0] =  alpha
            self.b[1] =  0.0
            self.b[2] = -alpha
            self.a[0] =  1.0 + alpha
            self.a[1] = -2.0 * cos_w0
            self.a[2] =  1.0 - alpha

        elif filter_type == "AllPass":
            self.b[0] =  1.0 - alpha
            self.b[1] = -2.0 * cos_w0
            self.b[2] =  1.0 + alpha
            self.a[0] =  1.0 + alpha
            self.a[1] = -2.0 * cos_w0
            self.a[2] =  1.0 - alpha

        elif filter_type == "Notch":
            self.b[0] =  1.0
            self.b[1] = -2.0 * cos_w0
            self.b[2] =  1.0
            self.a[0] =  1.0 + alpha
            self.a[1] = -2.0 * cos_w0
            self.a[2] =  1.0 - alpha

        elif filter_type == "Peaking":
            if A != 1.0:
                self.b[0] =  1.0 + alpha * A
                self.b[1] = -2.0 * cos_w0
                self.b[2] =  1.0 - alpha * A
                self.a[0] =  1.0 + alpha / A
                self.a[1] = -2.0 * cos_w0
                self.a[2] =  1.0 - alpha / A

        elif filter_type == "LowShelf":
            if A != 1.0:
                self.b[0] =     A * (A_add_1 - A_sub_1 * cos_w0 + 2 * sqrt_A * alpha)
                self.b[1] = 2 * A * (A_sub_1 - A_add_1 * cos_w0)
                self.b[2] =     A * (A_add_1 - A_sub_1 * cos_w0 - 2 * sqrt_A * alpha)
                self.a[0] =          A_add_1 + A_sub_1 * cos_w0 + 2 * sqrt_A * alpha
                self.a[1] =    -2 * (A_sub_1 + A_add_1 * cos_w0)
                self.a[2] =          A_add_1 + A_sub_1 * cos_w0 - 2 * sqrt_A * alpha

        elif filter_type == "HighShelf":
            if A != 1.0:
                self.b[0] =      A * (A_add_1 + A_sub_1 * cos_w0 + 2 * sqrt_A * alpha)
                self.b[1] = -2 * A * (A_sub_1 + A_add_1 * cos_w0)
                self.b[2] =      A * (A_add_1 + A_sub_1 * cos_w0 - 2 * sqrt_A * alpha)
                self.a[0] =           A_add_1 - A_sub_1 * cos_w0 + 2 * sqrt_A * alpha
                self.a[1] =      2 * (A_sub_1 - A_add_1 * cos_w0)
                self.a[2] =           A_add_1 - A_sub_1 * cos_w0 - 2 * sqrt_A * alpha
        else:
            raise ValueError(f"invalid filter_type: {filter_type}")
        
        self.b /= self.a[0]
        self.a /= self.a[0]
        
    def process(self, x):
        y = self.b[0] * x\
            + self.b[1] * self.x_buf[1]\
            + self.b[2] * self.x_buf[0]\
            - self.a[1] * self.y_buf[1]\
            - self.a[2] * self.y_buf[0]
        self.x_buf[0] = self.x_buf[1]
        self.x_buf[1] = x
        self.y_buf[0] = self.y_buf[1]
        self.y_buf[1] = y
        return y

    def dump(self):
        print("Biquad Type: %s, Fc: %.1fHz, Gain: %.1fdB"%(self.filter_type, self._fc, self._gain_db))
        print("       a0: %f, a1: %f, a2: %f"%(self.a[0], self.a[1], self.a[2]))
        print("       b0: %f, b1: %f, b2: %f\n"%(self.b[0], self.b[1], self.b[2]))