import argparse
import json
import math
import numpy as np

TICKS_PER_BEAT = 480.0

def hz_to_midi(hz: float) -> float:
    # A4=440Hz -> MIDI 69
    return 69.0 + 12.0 * math.log2(hz / 440.0)

def f0_to_pitch(f0: np.ndarray, fps: float, bpm: float, step: int, min_hz: float, hold_unvoiced: bool, pad_bars: float):
    f0 = f0.astype(np.float32)
    ticks_per_second = TICKS_PER_BEAT * bpm / 60.0
    total_seconds = len(f0) / fps
    total_ticks = total_seconds * ticks_per_second

    pitch_points = []
    last_midi = 60.0

    for i in range(0, len(f0), step):
        hz = float(f0[i])
        if (not math.isfinite(hz)) or hz < min_hz:
            if hold_unvoiced:
                m = last_midi
            else:
                continue
        else:
            m = hz_to_midi(hz)
            last_midi = m

        pos_tick = (i / fps) * ticks_per_second
        pitch_points.extend([round(pos_tick, 3), float(round(m, 6))])

    # 4/4 一小节 tick 数 = TICKS_PER_BEAT*4
    ticks_per_bar = TICKS_PER_BEAT * 4.0
    duration = float(round(total_ticks + ticks_per_bar * pad_bars, 3))

    return pitch_points, duration
