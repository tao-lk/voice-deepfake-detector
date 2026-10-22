# 声鉴 · 语音深度伪造检测工作台

基于 **Spectra-AASIST3**（INT8 量化 · ONNX Runtime CPU 推理）的本地离线语音深伪检测桌面应用。拖入音频或视频即可判定真伪，无需联网、无需安装 Python 环境（使用打包版时）。

![UI 预览](ui_preview.png)

## 功能特性

- **真伪判定**：输出 真实 BONAFIDE / 伪造 SPOOF 结论，附置信度与两类概率、单次推理耗时
- **多格式支持**：音频 wav/mp3/flac/m4a/aac/ogg/opus 等；视频 mp4/mkv/mov/avi 等，自动用 ffmpeg 分离音轨
- **波形视口 + 截取窗口**：视口滚动显示长波形；约 4 s 定长截取窗口（虚线边界）悬停变抓手、拖动选择送检片段、拖至边缘视口自动滚动、松手自动重检
- **暖调鉴定室 UI**：深琥珀主题，字号统一 2 倍缩放（`FONT_SCALE`），窗口自适应分栏
- **判定阈值校准**：默认 0.5 工作点准确率仅 86%，经两份数据集扫描改为 0.9999999658，准确率提升至 99%+

## 项目结构

```
.
├── app.py               # 主程序：UI + ffmpeg 解码 + ONNX 推理
├── eval_dataset.py      # 数据集批量评估：EER/AUC/多准则阈值扫描/分攻击类型统计
├── selfcheck.py         # offscreen UI 自检
├── build.ps1            # 一键打包：PyInstaller + 裁剪无用 DLL + 拷贝说明文档
├── build_app.spec       # PyInstaller 配置：onedir、无控制台、内嵌模型与 ffmpeg
├── 介绍.txt / 使用说明.txt  # 随包发布文档（UTF-8 BOM）
├── bin/                 # ffmpeg.exe / ffprobe.exe（gitignore）
└── models/
    └── spectra-aasist3-int8-audio-deepfake/
        └── spectra-aasist3-int8-dynamic.onnx   # 347 MB（gitignore）
```

## 模型契约

| 项目 | 值 |
|---|---|
| 输入 | `wav` float32 `[batch, 64600]` @ 16 kHz 单声道 |
| 输出 | `logits` `[batch, 2]` |
| 标签 | 0 = spoof 伪造，1 = bonafide 真实 |
| 预处理 | 预加重 0.97；长音频取截取窗口内 64600 样本，短音频循环平铺至满长 |
| 推理 | ONNX Runtime CPU EP，INT8 动态量化模型，首测加载约 5 s 后缓存 |

## 快速开始（开发环境）

依赖：Python 3.8+，`PyQt5` `numpy` `onnxruntime`（评估脚本另需 `soundfile` `loguru` `tqdm`）。ffmpeg/ffprobe 放入 `bin/`，模型放入 `models/` 对应子目录。

```powershell
python app.py          # 启动界面
python selfcheck.py    # offscreen UI 自检
```

## 打包发布

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

产物 `dist\声鉴\`（约 662 MB，含模型与 ffmpeg）：

- `声鉴.exe` — 无控制台窗口，双击运行
- `_internal/` — Python 运行时、PyQt5、onnxruntime、模型、ffmpeg
- `介绍.txt` / `使用说明.txt` — 自动随包拷贝

打包要点（`build_app.spec`）：

- `hiddenimports` 补 `secrets`（numpy.random Cython 动态导入，否则运行时报 `ModuleNotFoundError`）
- `excludes` 排除 QtWebEngine/Quick 等未用重型模块
- 打包后自动裁剪 `opengl32sw.dll`、`Qt5Quick/Qml`、`DirectML.dll` 等无用 DLL（省约 37 MB）

## 评估

`eval_dataset.py` 支持任意「真实子目录 + 多个伪造子目录」布局，输出总体指标、四准则最优阈值（EER 工作点 / 最高 Acc / 最高 F1 / 最大 Youden J）、UI 阈值下分攻击类型统计，并导出概率 `*_scores.npz`。

```powershell
# 自建数据集: dataset/{fake,real}/*.wav
python eval_dataset.py --dataset <路径>

# ASVspoof2019 子集: ys=原声, hc/hh/zh=变声伪造
python eval_dataset.py --dataset <路径> --real-dir ys --spoof-dirs hc,hh,zh --out-prefix eval_asv2019
```

### 实测结果

| 数据集 | 样本 | EER | AUC | Accuracy @UI阈值 |
|---|---|---|---|---|
| 本地标注集 | 1999 | 1.00% | 0.9985 | **99.15%**（FP=2 FN=15） |
| ASVspoof2019 子集 | 8000 | 0.75% | 0.9965 | 99.05%；hc/hh/zh 三类变声 **100% 检出** |

### 阈值校准说明

模型 softmax 输出高度饱和（真实样本概率集中在 1−1e−8 量级），默认阈值 0.5 恰处于饱和区中间，区分度差。对两份数据集联合扫描后，`app.py` 中判定阈值定为 `THRESHOLD_BONA = 0.9999999658`：

| 阈值 | 本地集 Acc | ASVspoof 子集 Acc |
|---|---|---|
| 0.5（默认） | 86.14% | 99.40% |
| 0.8988（ASV 最优点） | 89.04% | 99.50% |
| **0.9999999658（采用）** | **99.15%** | **99.05%** |

## 使用提示

- 每次判定仅取截取窗口内约 4 秒内容，长音频拖动窗口分段复检
- 首次检测加载模型需数秒，之后走缓存
- 结果仅供辅助参考，详见 `使用说明.txt`

## 许可与致谢

模型：Spectra-AASIST3（INT8 量化版）；评估参考 ASVspoof 2019 LA 数据集子集。
