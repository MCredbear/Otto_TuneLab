# Otto TuneLab

[English](README.md) | 简体中文

一个用来生成&优化 TuneLab 工程的工具箱。

## 功能

1. 用 [RMVPE](https://arxiv.org/abs/2306.15412v2) 从人声导出**音高线**。
2. 将 [SOFA](https://github.com/qiuqiao/SOFA) 或 [HubertFA](https://github.com/wolfgitpr/HubertFA) 生成的 **TextGrid** 转换为**音符**。（目前仅支持日语）
3. 通过对**音高线**求导来得到 **Gender 线**和 **Breathiness 线**。

## 使用方法

1. `git clone` 这个仓库.
2. 创建一个 python 环境然后运行 `pip install -r requirements_cpu.txt`。（RMVPE 模型很小，在 CPU 上也跑得够快了，当然你也可以改一下代码和 python 依赖来让它跑在 GPU 上）
3. 从[这里](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx)下载 `rmvpe.onnx` 并把它放在项目根目录里。
4. 运行 `python webUI.py` 然后在浏览器打开 `http://127.0.0.1:7860`。

## 鸣谢
- [TuneLab](https://github.com/LiuYunPlayer/TuneLab)
- [RMVPE](https://arxiv.org/abs/2306.15412v2)
  - 预训练模型由 [yxlllc](https://github.com/yxlllc/RMVPE) 和 [RVC-Boss](https://github.com/RVC-Boss) 完成训练和测试。
- [调用 RMVPE 的代码：RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/)
