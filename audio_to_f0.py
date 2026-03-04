import numpy as np
import soundfile as sf
import librosa
from infer.lib.rmvpe import RMVPE

RMVPE_PT = r"rmvpe.pt"
RMVPE_ONNX = r"rmvpe.onnx"
DEVICE = "privateuseone"
TARGET_SR = 16000

def audio_to_f0(input_path : str) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(input_path)
    info = sf.info(input_path)
    duration = info.frames / info.samplerate
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    model = RMVPE(RMVPE_ONNX, device=DEVICE, is_half=False, use_jit=False)

    f0 = model.infer_from_audio(audio)

    f0 = np.asarray(f0, dtype=np.float32)
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    f0[f0 < 1.0] = 0.0

    fps = len(f0) / duration
    print("duration (s):", duration)
    print("f0 frames:", len(f0))
    print("fps:", fps)
    return f0, fps