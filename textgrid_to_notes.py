import argparse, json, math
from typing import List, Dict
import numpy as np
import textgrid

TPB = 480.0

PH_MAP: Dict[str, List[str]] = {
    "u": ["M"],
    "ti": ["t'", "i"],
    "tsu": ["ts", "M"],
    "tu": ["t", "M"],
    "di": ["d'", "i"],
    "du": ["d'", "M"],
    "hi": ["C", "i"],
    "hu": ["b", "M"],
    "bi": ["b'", "i"],
    "bu": ["b", "M"],
    "fa": ["p\\", "a"],
    "fi": ["p\\", "i"],
    "fu": ["p\\", "M"],
    "fe": ["p\\", "e"],
    "fo": ["p\\", "o"],
    "pi": ["p'", "i"],
    "pu": ["p", "M"],
    "ni": ["J", "i"],
    "nu": ["n", "M"],
    "mi": ["m'", "i"],
    "mu": ["m", "M"],
    "ki": ["k'", "i"],
    "ku": ["k", "M"],
    "gi": ["g'", "i"],
    "gu": ["g", "M"],
    "ja": ["dZ", "a"],
    "ji": ["dZ", "i"],
    "ju": ["dZ", "M"],
    "je": ["dZ", "e"],
    "jo": ["dZ", "o"],
    "za": ["dz", "a"],
    "zi": ["dz", "i"],
    "zu": ["dz", "M"],
    "ze": ["dz", "e"],
    "zo": ["dz", "o"],
    "su": ["s", "M"],
    "sha": ["S", "a"],
    "shi": ["S", "i"],
    "shu": ["S", "M"],
    "she": ["S", "e"],
    "sho": ["S", "o"],
    "cha": ["tS", "a"],
    "chi": ["tS", "i"],
    "chu": ["tS", "M"],
    "che": ["tS", "e"],
    "cho": ["tS", "o"],
    "ra": ["4", "a"],
    "ri": ["4'", "i"],
    "ru": ["4", "M"],
    "re": ["4", "e"],
    "ro": ["4", "o"],
    "ya": ["j", "a"],
    "yu": ["j", "M"],
    "ye": ["j", "e"],
    "yo": ["j", "o"],
    "kya": ["k'", "a"],
    "kyu": ["k'", "M"],
    "kye": ["k'", "e"],
    "kyo": ["k'", "o"],
    "n": ["N"],
}

SKIP_DEFAULT = {"SP", "AP", "sil", "pau", ""}


def load_textgrid_intervals(path: str, word_tier: str, phone_tier: str):
    tg = textgrid.TextGrid.fromFile(path)
    wt = None
    pt = None
    for tier in tg:
        if getattr(tier, "name", None) == word_tier:
            wt = tier
        if getattr(tier, "name", None) == phone_tier:
            pt = tier
    if wt is None:
        raise ValueError(f"TextGrid 中未找到 word tier: {word_tier}")
    if pt is None:
        raise ValueError(f"TextGrid 中未找到 phone tier: {phone_tier}")

    words = [(float(iv.minTime), float(iv.maxTime), str(iv.mark)) for iv in wt]
    phones = [(float(iv.minTime), float(iv.maxTime), str(iv.mark)) for iv in pt]
    return words, phones


def sec_to_tick(sec: float, bpm: float) -> float:
    ticks_per_second = TPB * bpm / 60.0
    return sec * ticks_per_second


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def f0_hz_to_midi_series(f0: np.ndarray) -> np.ndarray:
    out = np.empty_like(f0, dtype=np.float32)
    last = 60.0
    for i, hz in enumerate(f0.astype(np.float32)):
        if (not np.isfinite(hz)) or hz < 1.0:
            out[i] = last
        else:
            m = hz_to_midi(float(hz))
            out[i] = m
            last = m
    return out


def collect_phones_in_window(phones, s_sec: float, e_sec: float, skip: set):
    out = []
    for ps, pe, lab in phones:
        lab = lab.strip()
        if lab in skip:
            continue
        if pe <= s_sec or ps >= e_sec:
            continue
        ps2 = max(ps, s_sec)
        pe2 = min(pe, e_sec)
        if pe2 > ps2:
            out.append((ps2 - s_sec, pe2 - s_sec, lab))  # 相对 syllable 起点
    return out


def map_syllable(raw_syms: List[str]) -> List[str]:
    key = "".join(raw_syms)
    return PH_MAP.get(key, raw_syms)


def textgrid_to_notes(
    textgrid_path: str,
    word_tier: str,
    phone_tier: str,
    bpm: float,
    fps: float,
    f0: np.ndarray,
    time_offset_sec: float = 0.0,
    skip_phones: set = SKIP_DEFAULT,
) -> List[Dict]:
    try:
        words, phones = load_textgrid_intervals(textgrid_path, word_tier, phone_tier)
    except Exception as e:
        raise ValueError(f"加载 TextGrid 文件时出错: {e}")
    midi = f0_hz_to_midi_series(f0)
    notes = []
    for ws, we, wlab in words:
        wlab = wlab.strip()
        if wlab in skip_phones:
            continue
        if we <= ws:
            continue

        syl_len = we - ws
        ws_global = ws + float(time_offset_sec)
        we_global = we + float(time_offset_sec)
        pos_tick = sec_to_tick(ws_global, bpm)
        dur_tick = sec_to_tick(syl_len, bpm)

        i0 = max(0, int(round(ws_global * fps)))
        i1 = min(len(midi), int(round(we_global * fps)))
        if i1 > i0 + 1:
            pitch_int = int(round(float(np.median(midi[i0:i1]))))

        raw = collect_phones_in_window(phones, ws, we, skip_phones)
        raw_syms = [x[2] for x in raw]
        mapped_syms = map_syllable(raw_syms)

        ph_list = []
        for (st, et, _raw_sym), sym2 in zip(raw, mapped_syms):
            ph_list.append(
                {
                    "startTime": float(st),
                    "endTime": float(et),
                    "symbol": sym2,
                }
            )

        note = {
            "pos": float(round(pos_tick, 3)),
            "dur": float(round(dur_tick, 3)),
            "pitch": int(pitch_int),
            "lyric": wlab,
            "pronunciation": "",
            "properties": {},
            "phonemes": ph_list,
        }
        notes.append(note)
    return notes
