import gradio as gr
from tabs.bpm_calc_tab import build_tab as build_bpm_calc_tab
from tabs.breathiness_gender_tab import build_tab as build_breathiness_gender_tab
from tabs.pitch_notes_tab import build_tab as build_pitch_notes_tab


with gr.Blocks(title="Otto TuneLab") as demo:
    gr.Markdown("**本项目目前只会操作工程文件的第一个轨道的第一个片段**")
    with gr.Tabs():
        build_pitch_notes_tab()
        build_breathiness_gender_tab()
        build_bpm_calc_tab()

if __name__ == "__main__":
    demo.launch()
