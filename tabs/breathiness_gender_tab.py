import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from pitch_derivative import (
    breathiness_from_pitch_derivative,
    gender_from_pitch_derivative,
)


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
    except Exception:
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


def build_tab():
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
