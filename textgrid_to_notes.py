import argparse, json, math
from enum import Enum
from typing import List, Dict
import numpy as np
import textgrid

TPB = 480.0


class LyricLanguage(str, Enum):
    JAPANESE = "ja"
    CHINESE = "zh"

JAPANESE_PH_MAP: Dict[str, List[str]] = {
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

CHINESE_PH_MAP: Dict[str, List[str]] = {
    "a": ["a"],
    "ai": ["aI"],
    "ao": ["AU"],
    "ba": ["p", "a"],
    "bai": ["p", "aI"],
    "ban": ["p", "a_n"],
    "bang": ["p", "AN"],
    "bao": ["p", "AU"],
    "bei": ["p", "ei"],
    "ben": ["p", "@_n"],
    "beng": ["p", "@N"],
    "bi": ["p", "i"],
    "bian": ["p", "iE_n"],
    "biao": ["p", "iAU"],
    "bie": ["p", "iE_r"],
    "bin": ["p", "i_n"],
    "bing": ["p", "iN"],
    "bo": ["p", "o"],
    "bu": ["p", "u"],
    "ca": ["ts_h", "a"],
    "cai": ["ts_h", "aI"],
    "can": ["ts_h", "a_n"],
    "cang": ["ts_h", "AN"],
    "cao": ["ts_h", "AU"],
    "ce": ["ts_h", "7"],
    "cen": ["ts_h", "@_n"],
    "ceng": ["ts_h", "@N"],
    "cha": ["ts`_h", "a"],
    "chai": ["ts`_h", "aI"],
    "chan": ["ts`_h", "a_n"],
    "chang": ["ts`_h", "AN"],
    "chao": ["ts`_h", "AU"],
    "che": ["ts`_h", "7"],
    "chen": ["ts`_h", "@_n"],
    "cheng": ["ts`_h", "@N"],
    "chi": ["ts`_h", "i`"],
    "chong": ["ts`_h", "UN"],
    "chou": ["ts`_h", "@U"],
    "chu": ["ts`_h", "u"],
    "chua": ["ts`_h", "ua"],
    "chuai": ["ts`_h", "uaI"],
    "chuan": ["ts`_h", "ua_n"],
    "chuang": ["ts`_h", "uAN"],
    "chui": ["ts`_h", "uei"],
    "chun": ["ts`_h", "u@_n"],
    "chuo": ["ts`_h", "uo"],
    "ci": ["ts_h", "i\\"],
    "cong": ["ts_h", "UN"],
    "cou": ["ts_h", "@U"],
    "cu": ["ts_h", "u"],
    "cuan": ["ts_h", "ua_n"],
    "cui": ["ts_h", "uei"],
    "cun": ["ts_h", "u@_n"],
    "cuo": ["ts_h", "uo"],
    "da": ["t", "a"],
    "dai": ["t", "aI"],
    "dan": ["t", "a_n"],
    "dang": ["t", "AN"],
    "dao": ["t", "AU"],
    "de": ["t", "7"],
    "dei": ["t", "ei"],
    "den": ["t", "@_n"],
    "deng": ["t", "@N"],
    "di": ["t", "i"],
    "dian": ["t", "iE_n"],
    "diao": ["t", "iAU"],
    "die": ["t", "iE_r"],
    "ding": ["t", "iN"],
    "diu": ["t", "i@U"],
    "dong": ["t", "UN"],
    "dou": ["t", "@U"],
    "du": ["t", "u"],
    "duan": ["t", "ua_n"],
    "dui": ["t", "uei"],
    "dun": ["t", "u@_n"],
    "duo": ["t", "uo"],
    "e": ["7"],
    "ei": ["ei"],
    "er": ["@`"],
    "fa": ["f", "a"],
    "fan": ["f", "a_n"],
    "fang": ["f", "AN"],
    "fei": ["f", "ei"],
    "fen": ["f", "@_n"],
    "feng": ["f", "@N"],
    "fo": ["f", "o"],
    "fou": ["f", "@U"],
    "fu": ["f", "u"],
    "ga": ["k", "a"],
    "gai": ["k", "aI"],
    "gan": ["k", "a_n"],
    "gang": ["k", "AN"],
    "gao": ["k", "AU"],
    "ge": ["k", "7"],
    "gei": ["k", "ei"],
    "gen": ["k", "@_n"],
    "geng": ["k", "@N"],
    "gong": ["k", "UN"],
    "gou": ["k", "@U"],
    "gu": ["k", "u"],
    "gua": ["k", "ua"],
    "guai": ["k", "uaI"],
    "guan": ["k", "ua_n"],
    "guang": ["k", "uAN"],
    "gui": ["k", "uei"],
    "gun": ["k", "u@_n"],
    "guo": ["k", "uo"],
    "ha": ["x", "a"],
    "hai": ["x", "aI"],
    "han": ["x", "a_n"],
    "hang": ["x", "AN"],
    "hao": ["x", "AU"],
    "he": ["x", "7"],
    "hei": ["x", "ei"],
    "hen": ["x", "@_n"],
    "heng": ["x", "@N"],
    "hong": ["x", "UN"],
    "hou": ["x", "@U"],
    "hu": ["x", "u"],
    "hua": ["x", "ua"],
    "huai": ["x", "uaI"],
    "huan": ["x", "ua_n"],
    "huang": ["x", "uAN"],
    "hui": ["x", "uei"],
    "hun": ["x", "u@_n"],
    "huo": ["x", "uo"],
    "i": ["i"],
    "ie": ["iE_r"],
    "iu": ["i@U"],
    "ji": ["ts\\", "i"],
    "jia": ["ts\\", "ia"],
    "jian": ["ts\\", "iE_n"],
    "jiang": ["ts\\", "iAN"],
    "jiao": ["ts\\", "iAU"],
    "jie": ["ts\\", "iE_r"],
    "jin": ["ts\\", "i_n"],
    "jing": ["ts\\", "iN"],
    "jiong": ["ts\\", "iUN"],
    "jiu": ["ts\\", "i@U"],
    "ju": ["ts\\", "y"],
    "juan": ["ts\\", "y{_n"],
    "jue": ["ts\\", "yE_r"],
    "jun": ["ts\\", "y_n"],
    "ka": ["k_h", "a"],
    "kai": ["k_h", "aI"],
    "kan": ["k_h", "a_n"],
    "kang": ["k_h", "AN"],
    "kao": ["k_h", "AU"],
    "ke": ["k_h", "7"],
    "ken": ["k_h", "@_n"],
    "keng": ["k_h", "@N"],
    "kong": ["k_h", "UN"],
    "kou": ["k_h", "@U"],
    "ku": ["k_h", "u"],
    "kua": ["k_h", "ua"],
    "kuai": ["k_h", "uaI"],
    "kuan": ["k_h", "ua_n"],
    "kuang": ["k_h", "uAN"],
    "kui": ["k_h", "uei"],
    "kun": ["k_h", "u@_n"],
    "kuo": ["k_h", "uo"],
    "la": ["l", "a"],
    "lai": ["l", "aI"],
    "lan": ["l", "a_n"],
    "lang": ["l", "AN"],
    "lao": ["l", "AU"],
    "le": ["l", "7"],
    "lei": ["l", "ei"],
    "leng": ["l", "@N"],
    "li": ["l", "i"],
    "lia": ["l", "ia"],
    "lian": ["l", "iE_n"],
    "liang": ["l", "iAN"],
    "liao": ["l", "iAU"],
    "lie": ["l", "iE_r"],
    "lin": ["l", "i_n"],
    "ling": ["l", "iN"],
    "liu": ["l", "i@U"],
    "lo": ["l", "o"],
    "long": ["l", "UN"],
    "lou": ["l", "@U"],
    "lu": ["l", "u"],
    "luan": ["l", "ua_n"],
    "lun": ["l", "u@_n"],
    "luo": ["l", "uo"],
    "lv": ["l", "y"],
    "lve": ["l", "yE_r"],
    "ma": ["m", "a"],
    "mai": ["m", "aI"],
    "man": ["m", "a_n"],
    "mang": ["m", "AN"],
    "mao": ["m", "AU"],
    "me": ["m", "7"],
    "mei": ["m", "ei"],
    "men": ["m", "@_n"],
    "meng": ["m", "@N"],
    "mi": ["m", "i"],
    "mian": ["m", "iE_n"],
    "miao": ["m", "iAU"],
    "mie": ["m", "iE_r"],
    "min": ["m", "i_n"],
    "ming": ["m", "iN"],
    "miu": ["m", "i@U"],
    "mo": ["m", "o"],
    "mou": ["m", "@U"],
    "mu": ["m", "u"],
    "na": ["n", "a"],
    "nai": ["n", "aI"],
    "nan": ["n", "a_n"],
    "nang": ["n", "AN"],
    "nao": ["n", "AU"],
    "ne": ["n", "7"],
    "nei": ["n", "ei"],
    "nen": ["n", "@_n"],
    "neng": ["n", "@N"],
    "ni": ["n", "i"],
    "nian": ["n", "iE_n"],
    "niang": ["n", "iAN"],
    "niao": ["n", "iAU"],
    "nie": ["n", "iE_r"],
    "nin": ["n", "i_n"],
    "ning": ["n", "iN"],
    "niu": ["n", "i@U"],
    "nong": ["n", "UN"],
    "nou": ["n", "@U"],
    "nu": ["n", "u"],
    "nuan": ["n", "ua_n"],
    "nun": ["a"],
    "nuo": ["n", "uo"],
    "nv": ["n", "y"],
    "nve": ["n", "yE_r"],
    "o": ["o"],
    "ou": ["@U"],
    "pa": ["p_h", "a"],
    "pai": ["p_h", "aI"],
    "pan": ["p_h", "a_n"],
    "pang": ["p_h", "AN"],
    "pao": ["p_h", "AU"],
    "pei": ["p_h", "ei"],
    "pen": ["p_h", "@_n"],
    "peng": ["p_h", "@N"],
    "pi": ["p_h", "i"],
    "pian": ["p_h", "iE_n"],
    "piao": ["p_h", "iAU"],
    "pie": ["p_h", "iE_r"],
    "pin": ["p_h", "i_n"],
    "ping": ["p_h", "iN"],
    "po": ["p_h", "o"],
    "pou": ["p_h", "@U"],
    "pu": ["p_h", "u"],
    "qi": ["ts\\_h", "i"],
    "qia": ["ts\\_h", "ia"],
    "qian": ["ts\\_h", "iE_n"],
    "qiang": ["ts\\_h", "iAN"],
    "qiao": ["ts\\_h", "iAU"],
    "qie": ["ts\\_h", "iE_r"],
    "qin": ["ts\\_h", "i_n"],
    "qing": ["ts\\_h", "iN"],
    "qiong": ["ts\\_h", "iUN"],
    "qiu": ["ts\\_h", "i@U"],
    "qu": ["ts\\_h", "y"],
    "quan": ["ts\\_h", "y{_n"],
    "que": ["ts\\_h", "yE_r"],
    "qun": ["ts\\_h", "y_n"],
    "ran": ["z`", "a_n"],
    "rang": ["z`", "AN"],
    "rao": ["z`", "AU"],
    "re": ["z`", "7"],
    "ren": ["z`", "@_n"],
    "reng": ["z`", "@N"],
    "ri": ["z`", "i`"],
    "rong": ["z`", "UN"],
    "rou": ["z`", "@U"],
    "ru": ["z`", "u"],
    "ruan": ["z`", "ua_n"],
    "rui": ["z`", "uei"],
    "run": ["z`", "u@_n"],
    "ruo": ["z`", "uo"],
    "sa": ["s", "a"],
    "sai": ["s", "aI"],
    "san": ["s", "a_n"],
    "sang": ["s", "AN"],
    "sao": ["s", "AU"],
    "se": ["s", "7"],
    "sen": ["s", "@_n"],
    "seng": ["s", "@N"],
    "sha": ["s`", "a"],
    "shai": ["s`", "aI"],
    "shan": ["s`", "a_n"],
    "shang": ["s`", "AN"],
    "shao": ["s`", "AU"],
    "she": ["s`", "7"],
    "shen": ["s`", "@_n"],
    "sheng": ["s`", "@N"],
    "shi": ["s`", "i`"],
    "shou": ["s`", "@U"],
    "shu": ["s`", "u"],
    "shua": ["s`", "ua"],
    "shuai": ["s`", "uaI"],
    "shuan": ["s`", "ua_n"],
    "shuang": ["s`", "uAN"],
    "shui": ["s`", "uei"],
    "shun": ["s`", "u@_n"],
    "shuo": ["s`", "uo"],
    "si": ["s", "i\\"],
    "song": ["s", "UN"],
    "sou": ["s", "@U"],
    "su": ["s", "u"],
    "suan": ["s", "ua_n"],
    "sui": ["s", "uei"],
    "sun": ["s", "u@_n"],
    "suo": ["s", "uo"],
    "ta": ["t_h", "a"],
    "tai": ["t_h", "aI"],
    "tan": ["t_h", "a_n"],
    "tang": ["t_h", "AN"],
    "tao": ["t_h", "AU"],
    "te": ["t_h", "7"],
    "teng": ["t_h", "@N"],
    "ti": ["t_h", "i"],
    "tian": ["t_h", "iE_n"],
    "tiao": ["t_h", "iAU"],
    "tie": ["t_h", "iE_r"],
    "ting": ["t_h", "iN"],
    "tong": ["t_h", "UN"],
    "tou": ["t_h", "@U"],
    "tu": ["t_h", "u"],
    "tuan": ["t_h", "ua_n"],
    "tui": ["t_h", "uei"],
    "tun": ["t_h", "u@_n"],
    "tuo": ["t_h", "uo"],
    "u": ["u"],
    "ui": ["uei"],
    "v": ["y"],
    "ve": ["yE_r"],
    "wa": ["ua"],
    "wai": ["uaI"],
    "wan": ["ua_n"],
    "wang": ["uAN"],
    "wei": ["uei"],
    "wen": ["u@_n"],
    "weng": ["u@N"],
    "wo": ["uo"],
    "wu": ["u"],
    "xi": ["s\\", "i"],
    "xia": ["s\\", "ia"],
    "xian": ["s\\", "iE_n"],
    "xiang": ["s\\", "iAN"],
    "xiao": ["s\\", "iAU"],
    "xie": ["s\\", "iE_r"],
    "xin": ["s\\", "i_n"],
    "xing": ["s\\", "iN"],
    "xiong": ["s\\", "iUN"],
    "xiu": ["s\\", "i@U"],
    "xu": ["s\\", "y"],
    "xuan": ["s\\", "y{_n"],
    "xue": ["s\\", "yE_r"],
    "xun": ["s\\", "y_n"],
    "ya": ["ia"],
    "yan": ["iE_n"],
    "yang": ["iAN"],
    "yao": ["iAU"],
    "ye": ["iE_r"],
    "yi": ["i"],
    "yin": ["i_n"],
    "ying": ["iN"],
    "yong": ["iUN"],
    "you": ["i@U"],
    "yu": ["y"],
    "yuan": ["y{_n"],
    "yue": ["yE_r"],
    "yun": ["y_n"],
    "za": ["ts", "a"],
    "zai": ["ts", "aI"],
    "zan": ["ts", "a_n"],
    "zang": ["ts", "AN"],
    "zao": ["ts", "AU"],
    "ze": ["ts", "7"],
    "zei": ["ts", "ei"],
    "zen": ["ts", "@_n"],
    "zeng": ["ts", "@N"],
    "zha": ["ts`", "a"],
    "zhai": ["ts`", "aI"],
    "zhan": ["ts`", "a_n"],
    "zhang": ["ts`", "AN"],
    "zhao": ["ts`", "AU"],
    "zhe": ["ts`", "7"],
    "zhen": ["ts`", "@_n"],
    "zheng": ["ts`", "@N"],
    "zhi": ["ts`", "i`"],
    "zhong": ["ts`", "UN"],
    "zhou": ["ts`", "@U"],
    "zhu": ["ts`", "u"],
    "zhua": ["ts`", "ua"],
    "zhuai": ["ts`", "uaI"],
    "zhuan": ["ts`", "ua_n"],
    "zhuang": ["ts`", "uAN"],
    "zhui": ["ts`", "uei"],
    "zhun": ["ts`", "u@_n"],
    "zhuo": ["ts`", "uo"],
    "zi": ["ts", "i\\"],
    "zong": ["ts", "UN"],
    "zou": ["ts", "@U"],
    "zu": ["ts", "u"],
    "zuan": ["ts", "ua_n"],
    "zui": ["ts", "uei"],
    "zun": ["ts", "u@_n"],
    "zuo": ["ts", "uo"],
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


def map_japanese_syllable(raw_syms: List[str]) -> List[str]:
    key = "".join(raw_syms)
    return JAPANESE_PH_MAP.get(key, raw_syms)


def map_chinese_syllable(raw_syms: List[str]) -> List[str]:
    key = "".join(raw_syms)
    return CHINESE_PH_MAP.get(key, raw_syms)


def textgrid_to_notes(
    textgrid_path: str,
    word_tier: str,
    phone_tier: str,
    bpm: float,
    fps: float,
    f0: np.ndarray,
    time_offset_sec: float = 0.0,
    skip_phones: set = SKIP_DEFAULT,
    language: LyricLanguage | str = LyricLanguage.JAPANESE,
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
        lang = (
            language.value if isinstance(language, LyricLanguage) else str(language)
        )
        if lang == LyricLanguage.CHINESE.value:
            mapped_syms = map_chinese_syllable(raw_syms)
        else:
            mapped_syms = map_japanese_syllable(raw_syms)

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
