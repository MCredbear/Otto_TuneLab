import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import librosa

from audio_to_f0 import audio_to_f0
from f0_to_pitch import f0_to_pitch
from textgrid_to_notes import textgrid_to_notes


def _default_output_filename(input_path: str) -> str:
    if not input_path:
        return ""
    base, _ = os.path.splitext(os.path.basename(input_path))
    return f"{base}.tlp"


def sync_paths(audio_path: str):
    if not audio_path:
        return ""
    return _default_output_filename(audio_path)


def convert_to_tlp(
    input_path: str,
    output_filename: str,
    bpm: float,
    step: int,
    pad_bars: float,
    textgrid_paths,
    segment_audio_paths,
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
    if textgrid_paths:
        textgrid_list = [path for path in textgrid_paths if path]
        segment_audio_list = [path for path in (segment_audio_paths or []) if path]

        if len(segment_audio_list) != len(textgrid_list):
            return "TextGrid 数量与音频片段数量不一致。", None

        offset_sec = 0.0
        for tg_path, seg_audio_path in zip(textgrid_list, segment_audio_list):
            try:
                notes.extend(
                    textgrid_to_notes(
                        tg_path,
                        "words",
                        "phones",
                        bpm,
                        fps,
                        f0,
                        time_offset_sec=offset_sec,
                    )
                )
            except Exception as e:
                return e, None

            try:
                seg_duration = float(librosa.get_duration(path=seg_audio_path))
            except Exception as e:
                return f"读取音频片段时长失败：{seg_audio_path}，错误：{e}", None
            offset_sec += seg_duration

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


def build_tab():
    with gr.TabItem("音高线提取 & 音符插入"):
        gr.Markdown(
            """
上传人声干声，然后由 RMVPE 提取音高线再生成 TuneLab 项目文件（.tlp）。  
    可选上传多个 TextGrid 和对应音频片段：会按上一个音频片段时长累计偏移，插入歌词和音符。
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
            bpm = gr.Number(label="BPM", value=120.0)

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
            label="可选：TextGrid 文件（支持多选）",
            file_types=[".textgrid"],
            file_count="multiple",
        )

        segment_audio_input = gr.File(
            label="可选：对应音频片段（支持多选，顺序需与 TextGrid 一致）",
            file_types=[".wav", ".mp3", ".flac", ".m4a", ".ogg"],
            file_count="multiple",
        )

        with gr.Row():
            run_btn_tab_1 = gr.Button("开始生成", variant="primary")
        status_tab_1 = gr.Textbox(label="状态", interactive=False)
        output_file_tab_1 = gr.File(label="输出文件", interactive=False)

        audio_input.change(
            sync_paths, inputs=audio_input, outputs=[output_filename]
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
                segment_audio_input,
            ],
            outputs=[status_tab_1, output_file_tab_1],
        )
