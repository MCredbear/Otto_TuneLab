import json
import os
import tempfile
from pathlib import Path

import librosa
import gradio as gr

from audio_to_f0 import audio_to_f0
from f0_to_pitch import f0_to_pitch
from textgrid_to_notes import textgrid_to_notes
from pitch_derivative import (
    gender_from_pitch_derivative,
    breathiness_from_pitch_derivative,
)


def _default_output_filename(input_path: str) -> str:
    if not input_path:
        return ""
    base, _ = os.path.splitext(os.path.basename(input_path))
    return f"{base}.tlp"


def sync_paths(audio_path: str):
    if not audio_path:
        return ""
    return _default_output_filename(audio_path)


def toggle_bpm_input(auto_bpm: bool):
    return gr.update(interactive=not auto_bpm)


def calculate_bpm(audio_path: str, auto_bpm: bool):
    if not auto_bpm or not audio_path:
        return gr.update(), ""
    try:
        y, sr = librosa.load(audio_path, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo[0]), f"已自动估算 BPM: {float(tempo[0]):.2f}"
    except Exception as e:
        return gr.update(), f"自动估算 BPM 失败：{e}"


def convert_to_tlp(
    input_path: str,
    output_filename: str,
    bpm: float,
    step: int,
    pad_bars: float,
    textgrid_path: str,
):
    if output_filename is None or output_filename.strip() == "":
        return "输出文件名为空。", None

    if bpm is None or bpm <= 0:
        return "BPM 必须大于 0。", None

    f0, fps = audio_to_f0(input_path)
    pitch_points, duration = f0_to_pitch(
        f0,
        bpm=float(bpm),
        fps=fps,
        step=int(step),
        hold_unvoiced=True,
        min_hz=1.0,
        pad_bars=float(pad_bars),
    )

    notes = []
    if textgrid_path:
        try:
            notes = textgrid_to_notes(
                textgrid_path, "words", "phones", bpm, fps, f0
            )
        except Exception as e:
            return e, None

    proj = {
        "version": 0,
        "tempos": [{"pos": 0.0, "bpm": float(bpm)}],
        "timeSignatures": [{"barIndex": 0, "numerator": 4, "denominator": 4}],
        "tracks": [
            {
                "name": "轨道_1",
                "gain": 0.0,
                "pan": 0.0,
                "mute": False,
                "solo": False,
                "color": "#737CE5",
                "asRefer": True,
                "parts": [
                    {
                        "name": "片段_1",
                        "pos": 0.0,
                        "dur": duration,
                        "type": "midi",
                        "gain": 0.0,
                        "voice": {"type": "", "id": ""},
                        "properties": {},
                        "notes": notes,
                        "automations": {},
                        "pitch": [pitch_points],
                        "vibratos": [],
                    }
                ],
            }
        ],
    }

    out_dir = Path(tempfile.mkdtemp())
    out_path = out_dir / output_filename
    out_path.write_text(
        json.dumps(proj, ensure_ascii=False, indent=4), encoding="utf-8"
    )

    return f"完成：{output_filename}（points: {len(pitch_points)//2}）", str(out_path)


def calculate_gender_and_breathiness(
    tlp_path: str,
    is_calculate_gender: bool,
    gender_scale: float,
    gender_clamp: float,
    is_calculate_breathiness: bool,
    breathiness_scale: float,
    breathiness_clamp: float,
):
    output_filename = os.path.basename(tlp_path)
    with open(tlp_path, "r", encoding="utf-8") as f:
        proj = json.load(f)
    try:
        track0 = proj["tracks"][0]
        part0 = track0["parts"][0]
    except Exception as e:
        return "工程文件错误，检查第一个轨道的第一个片段是否存在", None

    pitch_points = part0["pitch"][0]

    if is_calculate_gender:
        gender_vals = gender_from_pitch_derivative(
            pitch_points, scale=gender_scale, clamp=gender_clamp
        )
        part0["automations"]["Gender"] = {"default": 0.0, "values": gender_vals}

    if is_calculate_breathiness:
        breathiness_vals = breathiness_from_pitch_derivative(
            pitch_points, scale=breathiness_scale, clamp=breathiness_clamp
        )
        part0["automations"]["Breathiness"] = {
            "default": 0.0,
            "values": breathiness_vals,
        }

    out_dir = Path(tempfile.mkdtemp())
    out_path = out_dir / output_filename
    out_path.write_text(
        json.dumps(proj, ensure_ascii=False, indent=4), encoding="utf-8"
    )

    return f"完成：{output_filename}", str(out_path)


with gr.Blocks(title="Otto TuneLab") as demo:
    gr.Markdown("**本项目目前只会操作工程文件的第一个轨道的第一个片段**")
    with gr.Tabs():
        with gr.TabItem("音高线提取 & 音符插入"):
            gr.Markdown(
                """
上传人声干声，然后由 RMVPE 提取音高线再生成 TuneLab 项目文件（.tlp）。  
可选上传 SOFA 或 HubertFA 生成的 TextGrid 文件以插入歌词和音符。
"""
            )

            audio_input = gr.Audio(
                label="音频文件（拖拽或点击上传）",
                type="filepath",
                sources=["upload"],
            )

            with gr.Row():
                output_filename = gr.Textbox(
                    label="输出 .tlp 文件名", placeholder="output.tlp"
                )

            with gr.Row():
                is_auto_bpm = gr.Checkbox(label="自动计算 BPM", value=True)
                bpm = gr.Number(label="BPM", value=120.0, interactive=False)

            step = gr.Number(
                label="每 N 帧计算一个音高点（默认 1，1 帧 = 10ms）",
                value=1,
                precision=0,
                minimum=1,
            )

            pad_bars = gr.Number(
                label="结尾 padding 小节数（默认 1）", value=1.0, minimum=0
            )

            textgrid_input = gr.File(
                label="可选：TextGrid 文件（拖拽或点击上传）", file_types=[".textgrid"]
            )

            with gr.Row():
                run_btn_tab_1 = gr.Button("开始生成", variant="primary")
            status_tab_1 = gr.Textbox(label="状态", interactive=False)
            output_file_tab_1 = gr.File(label="输出文件", interactive=False)

            audio_input.change(
                sync_paths, inputs=audio_input, outputs=[output_filename]
            )
            audio_input.change(
                calculate_bpm,
                inputs=[audio_input, is_auto_bpm],
                outputs=[bpm, status_tab_1],
            )
            is_auto_bpm.change(toggle_bpm_input, inputs=is_auto_bpm, outputs=bpm)
            is_auto_bpm.change(
                calculate_bpm,
                inputs=[audio_input, is_auto_bpm],
                outputs=[bpm, status_tab_1],
            )

            run_btn_tab_1.click(
                convert_to_tlp,
                inputs=[
                    audio_input,
                    output_filename,
                    bpm,
                    step,
                    pad_bars,
                    textgrid_input,
                ],
                outputs=[status_tab_1, output_file_tab_1],
            )
        with gr.TabItem("Breathiness & Gender 自动生成"):
            gr.Markdown(
                """
RMVPE 提取的音高线会有瑕疵，请手动修正后上传工程文件（.tlp），然后会计算音高线的导数来作为 Breathiness 和 Gender。
"""
            )
            tlp_input = gr.File(label="上传 .tlp 文件", file_types=[".tlp"])

            with gr.Row():
                is_calculate_gender = gr.Checkbox(label="计算 Gender", value=True)
                gender_scale = gr.Number(
                    label="Gender 强度",
                    minimum=0.0,
                    maximum=20.0,
                    value=5.0,
                    step=0.1,
                )
                gender_clamp = gr.Number(
                    label="Gender 最大值",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.01,
                )

            with gr.Row():
                is_calculate_breathiness = gr.Checkbox(
                    label="计算 Breathiness", value=True
                )
                breathiness_scale = gr.Number(
                    label="Breathiness 强度",
                    minimum=0.0,
                    maximum=20.0,
                    value=1.0,
                    step=0.1,
                )
                breathiness_clamp = gr.Number(
                    label="Breathiness 最大值",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.35,
                    step=0.01,
                )

            with gr.Row():
                run_btn_tab_2 = gr.Button("开始生成", variant="primary")
            status_tab_2 = gr.Textbox(label="状态", interactive=False)
            output_file_tab_2 = gr.File(label="输出文件", interactive=False)

            run_btn_tab_2.click(
                calculate_gender_and_breathiness,
                inputs=[
                    tlp_input,
                    is_calculate_gender,
                    gender_scale,
                    gender_clamp,
                    is_calculate_breathiness,
                    breathiness_scale,
                    breathiness_clamp,
                ],
                outputs=[status_tab_2, output_file_tab_2],
            )

if __name__ == "__main__":
    demo.launch()
