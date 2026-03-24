import gradio as gr
import librosa


def calculate_bpm(audio_path: str):
    if not audio_path:
        return None, "请先上传音频文件。"
    try:
        y, sr = librosa.load(audio_path, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo[0]), f"已自动估算 BPM: {float(tempo[0]):.2f}"
    except Exception as e:
        return None, f"自动估算 BPM 失败：{e}"


def build_tab():
    with gr.TabItem("BPM 自动计算"):
        gr.Markdown(
            """
上传音频文件后，点击按钮自动估算 BPM。
"""
        )

        bpm_audio_input = gr.Audio(
            label="音频文件（拖拽或点击上传）",
            type="filepath",
            sources=["upload"],
        )
        calc_bpm_btn = gr.Button("计算 BPM", variant="primary")
        bpm_result = gr.Number(label="估算 BPM", interactive=False)
        bpm_status = gr.Textbox(label="状态", interactive=False)

        calc_bpm_btn.click(
            calculate_bpm,
            inputs=[bpm_audio_input],
            outputs=[bpm_result, bpm_status],
        )
