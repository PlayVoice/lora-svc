# https://github.com/chenwj1989/pafx
import numpy as np
import soundfile as sf

from .comb import Comb
from .allpass import Allpass
from .tapped_delay_line import TappedDelayLine
from .reverb import Reverb, ReverbConfig
from .equalizer import Equalizer
from .echo import Echo
from .chorus import Chorus
from .flanger import Flanger
from .vibrato import Vibrato
from .tremolo import Tremolo
from .fade import Fade, FadeIn, FadeOut


def svc_eq(x):
    eq_gains = [-30, -10, 10, 10, 5, 0, -5, -10, -10, -10]
    eq = Equalizer(eq_gains, sample_rate=16000)
    eq.dump()
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = eq.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)

def svc_reverb(x):
    config = ReverbConfig()
    config.room_scale = 20
    config.pre_delay = 10
    config.dry_gain = -5
    config.wet_gain = 5
    config.hf_damping = 30
    config.reverberance = 30
    config.stereo_width = 10
    config.er_gain = 0.2
    reverb = Reverb(config, sample_rate=16000)
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = reverb.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


def svc_echo(x):
    echo_gains = [0.5]
    echo_delays = [0.05]
    echo = Echo(16000, echo_delays, echo_gains, 0.5)
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = echo.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


def svc_chorus(x):
    gains = [0.5]
    delays = [0.05]
    mod_widths = [0.005]
    mod_freqs = [2]
    dry_gain = 1
    chorus = Chorus(16000, delays, mod_freqs, mod_widths, gains, dry_gain)
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = chorus.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


def svc_flanger(x):
    delay = 0.01
    mod_width = 0.003
    mod_freq = 1
    flanger = Flanger(16000, delay, mod_width, mod_freq)
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = flanger.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


def svc_vibrato(x):
    delay = 0.008
    mod_width = 0.004
    mod_freq = 2.3
    vibrato = Vibrato(16000, delay, mod_width, mod_freq)
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = vibrato.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


def svc_tremolo(x):
    mod_depth = 0.5
    mod_freq = 10
    tremolo = Tremolo(mod_freq, mod_depth, 16000)
    # Start Processing
    y = np.zeros(len(x))
    x = x / 32768.0
    for i in range(len(x)):
        y[i] = tremolo.process(x[i])
    y = y / max(np.abs(y))
    y = y * 32768.0
    return y.astype(np.int16)


def svc_fade(input_file):
    x, fs = sf.read(input_file)
    y = np.zeros(len(x))
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))

    in_length = 5
    out_start = 5
    out_length = 5
    fade = Fade(fs, in_length, out_start, out_length)
    fade_in = FadeIn(fs, in_length)
    fade_out = FadeOut(fs, out_length)
    # Start Processing
    for i in range(len(x)):
        y[i] = fade.process(x[i])
        y1[i] = fade_in.process(x[i])
        y2[i] = fade_out.process(x[i])
    y = y / max(np.abs(y))
    y1 = y1 / max(np.abs(y1))
    y2 = y2 / max(np.abs(y2))
    sf.write("data/fade.wav", y, fs)
    sf.write("data/fade_in.wav", y1, fs)
    sf.write("data/fade_out.wav", y2, fs)


def svc_filters(input_file):
    x, fs = sf.read(input_file)
    comb = Comb(100, 0.9, 0)
    allpass = Allpass(556, 0.5)

    tap_delays = [190,  949,  993,  1183, 1192, 1315,
                  2021, 2140, 2524, 2590, 2625, 2700,
                  3119, 3123, 3202, 3268, 3321, 3515]

    tap_gains = [.841, .504, .49,  .379, .38,  .346,
                 .289, .272, .192, .193, .217,  .181,
                 .18,  .181, .176, .142, .167, .134]

    tdl = TappedDelayLine(tap_delays, tap_gains)

    yc = np.zeros(len(x))
    ya = np.zeros(len(x))
    yt = np.zeros(len(x))
    # Start Processing
    for i in range(len(x)):
        yc[i] = comb.process(x[i])
        ya[i] = allpass.process(x[i])
        yt[i] = tdl.process(x[i])

    # Save Results
    output_file = "data/comb.wav"
    yc = yc / max(np.abs(yc))
    sf.write(output_file, yc, fs)

    output_file = "data/allpass.wav"
    ya = ya / max(np.abs(ya))
    sf.write(output_file, ya, fs)

    output_file = "data/tpl.wav"
    yt = yt / max(np.abs(yt))
    sf.write(output_file, yt, fs)
