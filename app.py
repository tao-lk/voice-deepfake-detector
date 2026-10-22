# -*- coding: utf-8 -*-
"""声鉴 · Spectra-AASIST3 语音深度伪造检测工作台

上传音频或视频样例，自动提取 16kHz 单声道 PCM，
基于 ONNX INT8 量化模型进行真伪推理，可视化波形与判定结果。
"""

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QPen, QLinearGradient, QPainterPath, QBrush, QIcon, QPixmap
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFrame,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter, QFileDialog,
    QProgressBar, QMessageBox, QSizePolicy
)

# ---------------------------------------------------------------- 路径与常量

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spectra-aasist3-int8-audio-deepfake" / "spectra-aasist3-int8-dynamic.onnx"
FFMPEG = BASE_DIR / "bin" / "ffmpeg.exe"
FFPROBE = BASE_DIR / "bin" / "ffprobe.exe"
if not FFMPEG.exists():
    FFMPEG = Path("ffmpeg")
if not FFPROBE.exists():
    FFPROBE = Path("ffprobe")

SAMPLE_RATE = 16000
REQUIRED_SAMPLES = 64600
PREEMPH = 0.97
MODEL_WINDOW_S = REQUIRED_SAMPLES / SAMPLE_RATE  # ≈ 4.0375 s

# 判定阈值: 在 1999 条本地标注数据上扫描所得最优工作点
# Acc 99.15%, F1 0.9914, FP=2, FN=15; 默认 0.5 时 Acc 仅 86.14%
# 模型 softmax 高度饱和, 真实样本概率集中在 1-1e-8 附近, 故阈值接近 1
THRESHOLD_BONA = 0.9999999658

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".amr"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts", ".wmv", ".mpg", ".mpeg", ".3gp"}
ALL_EXTS = sorted(AUDIO_EXTS | VIDEO_EXTS)

FILE_FILTER = (
    "媒体文件 (*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.opus *.wma *.aiff *.amr "
    "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v *.ts *.wmv *.mpg *.mpeg *.3gp);;全部文件 (*)"
)

CREATE_NO_WINDOW = 0x08000000

# ---------------------------------------------------------------- 配色（暖调鉴定室）

C_BG = "#16130f"
C_CARD = "#1e1a15"
C_CARD2 = "#221d16"
C_BORDER = "#352e24"
C_BORDER_SOFT = "#2a241c"
C_TEXT = "#f2e8d5"
C_DIM = "#a3927a"
C_FAINT = "#6e6152"
C_AMBER = "#e0a458"
C_AMBER_HI = "#f0bc72"
C_ON_AMBER = "#1a140c"
C_BONA = "#9cbf9a"
C_SPOOF = "#d97761"
C_WARN = "#c9b48a"

# 字体统一缩放系数（所有界面字号 × 此值）
FONT_SCALE = 2


def recommend_duration(avail_mb: int):
    """依据当前可用内存估算安全的样例时长上限。"""
    reserve_mb = 1024
    usable = max(avail_mb - reserve_mb, 128)
    est = int(usable / 8)
    hi = min(max(est, 15), 180)
    return 5, hi


def get_available_ram_mb() -> int:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return 4096
    return int(stat.ullAvailPhys // (1 << 20))


# ---------------------------------------------------------------- 媒体处理

def probe_media(path: Path) -> dict:
    cmd = [
        str(FFPROBE), "-v", "error",
        "-show_entries", "format=duration,format_name,bit_rate",
        "-of", "default=noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.run(
        cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=30
    ).stdout.decode("utf-8", "replace")
    info = {"duration": None, "format_name": "", "bit_rate": ""}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "duration":
            try:
                info["duration"] = float(v)
            except ValueError:
                pass
        elif k == "format_name":
            info["format_name"] = v
        elif k == "bit_rate":
            info["bit_rate"] = v
    return info


def extract_audio(path: Path) -> np.ndarray:
    cmd = [
        str(FFMPEG), "-v", "error", "-nostdin",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-500:] or "ffmpeg 解码失败")
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise RuntimeError("未从文件中提取到音频轨道")
    return audio


def preemphasis(x: np.ndarray, c: float = PREEMPH) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - c * x[:-1]
    return y


def make_model_input(audio: np.ndarray) -> np.ndarray:
    y = preemphasis(audio.astype(np.float32, copy=False))
    n = y.shape[0]
    if n == 0:
        raise RuntimeError("空音频")
    if n < REQUIRED_SAMPLES:
        reps = REQUIRED_SAMPLES // n + 1
        y = np.tile(y, reps)
    return y[:REQUIRED_SAMPLES].astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def softmax_batch(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- 模型单例

_SESSION = None
_SESSION_LOAD_MS = 0.0


def get_session():
    global _SESSION, _SESSION_LOAD_MS
    if _SESSION is None:
        import onnxruntime as ort
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        t0 = time.perf_counter()
        _SESSION = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
        _SESSION_LOAD_MS = (time.perf_counter() - t0) * 1000.0
    return _SESSION, _SESSION_LOAD_MS


# ---------------------------------------------------------------- 工作线程

class DetectWorker(QThread):
    stage = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path

    def run(self):
        try:
            path = Path(self.file_path)
            self.stage.emit("正在探测媒体信息…")
            media = probe_media(path)

            self.stage.emit("正在提取 16 kHz 单声道音频…")
            audio = extract_audio(path)

            need_load = _SESSION is None
            self.stage.emit("正在加载模型，请稍候…" if need_load else "正在推理…")
            sess, load_ms = get_session()

            x = make_model_input(audio)
            self.stage.emit("正在推理…")
            t0 = time.perf_counter()
            logits = sess.run(None, {"wav": x[None, :]})[0][0]
            infer_ms = (time.perf_counter() - t0) * 1000.0

            probs = softmax(logits.astype(np.float64))
            p_spoof, p_bona = float(probs[0]), float(probs[1])
            verdict = "bonafide" if p_bona >= THRESHOLD_BONA else "spoof"
            confidence = p_bona if verdict == "bonafide" else p_spoof

            self.finished_ok.emit({
                "audio": audio,
                "media": media,
                "p_spoof": p_spoof,
                "p_bona": p_bona,
                "verdict": verdict,
                "confidence": confidence,
                "infer_ms": infer_ms,
                "load_ms": load_ms if need_load else 0.0,
                "first_load": need_load,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


# ---------------------------------------------------------------- 波形控件

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio = None
        self.peaks_min = None
        self.peaks_max = None
        self.setMinimumHeight(int(150 * FONT_SCALE))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_audio(self, audio: np.ndarray):
        self.audio = audio
        self._compute_peaks()
        self.update()

    def _compute_peaks(self, n: int = 1600):
        x = self.audio
        if x is None or x.size == 0:
            self.peaks_min = self.peaks_max = None
            return
        chunk = max(1, x.shape[0] // n)
        usable = x.shape[0] // chunk * chunk
        m = x[:usable].reshape(-1, chunk)
        self.peaks_min = m.min(axis=1)
        self.peaks_max = m.max(axis=1)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C_CARD2))

        pad_l, pad_r, pad_t, pad_b = 14, 14, int(26 * FONT_SCALE), int(24 * FONT_SCALE)
        rect = QRectF(pad_l, pad_t, w - pad_l - pad_r, h - pad_t - pad_b)
        if rect.height() < 10:
            p.end()
            return

        mid_y = rect.center().y()

        if self.audio is None or self.audio.size == 0:
            p.setPen(QPen(QColor(C_FAINT)))
            p.setFont(QFont("Microsoft YaHei UI", int(10 * FONT_SCALE)))
            p.drawText(self.rect(), Qt.AlignCenter, "尚未载入音频")
            p.end()
            return

        total = self.audio.shape[0]
        win_end = min(total, REQUIRED_SAMPLES)
        frac = win_end / total

        # 分析窗口高亮
        win_w = rect.width() * frac
        hl = QColor(C_AMBER)
        hl.setAlpha(28)
        p.fillRect(QRectF(rect.left(), rect.top() - 6, win_w, rect.height() + 12), hl)
        p.setPen(QPen(QColor(C_AMBER), 1, Qt.DashLine))
        p.drawLine(QPointF(rect.left() + win_w, rect.top() - 6),
                   QPointF(rect.left() + win_w, rect.bottom() + 6))

        # 中线
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.drawLine(QPointF(rect.left(), mid_y), QPointF(rect.right(), mid_y))

        # 包络
        n = self.peaks_min.shape[0]
        peak = float(max(np.abs(self.peaks_max).max(), np.abs(self.peaks_min).max(), 1e-6))
        half = rect.height() / 2 - 2

        grad = QLinearGradient(0, rect.top(), 0, rect.bottom())
        grad.setColorAt(0.0, QColor(C_AMBER_HI))
        grad.setColorAt(0.5, QColor(C_AMBER))
        grad.setColorAt(1.0, QColor("#b87e3c"))

        path = QPainterPath()
        path.moveTo(rect.left(), mid_y)
        for i in range(n):
            x = rect.left() + rect.width() * i / max(n - 1, 1)
            y = mid_y - (self.peaks_max[i] / peak) * half
            path.lineTo(x, y)
        for i in range(n - 1, -1, -1):
            x = rect.left() + rect.width() * i / max(n - 1, 1)
            y = mid_y - (self.peaks_min[i] / peak) * half
            path.lineTo(x, y)
        path.closeSubpath()
        p.fillPath(path, QBrush(grad))

        # 边框
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 10, 10)

        # 标注
        p.setFont(QFont("Microsoft YaHei UI", int(8 * FONT_SCALE)))
        p.setPen(QPen(QColor(C_AMBER)))
        p.drawText(QPointF(rect.left() + 6, rect.top() - int(8 * FONT_SCALE)),
                   f"分析窗口 0–{MODEL_WINDOW_S:.2f}s")
        dur = total / SAMPLE_RATE
        p.setPen(QPen(QColor(C_DIM)))
        p.drawText(QRectF(rect.left(), h - int(20 * FONT_SCALE), rect.width() - 60, int(16 * FONT_SCALE)),
                   Qt.AlignRight, f"{dur:.2f} s")
        p.drawText(QPointF(rect.left() + 6, h - int(8 * FONT_SCALE)), "16 kHz · mono · PCM")
        p.end()


# ---------------------------------------------------------------- 拖放区

class DropZone(QWidget):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(int(92 * FONT_SCALE))
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pick_file()

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择音频或视频样本", "", FILE_FILTER)
        if path:
            self.file_dropped.emit(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for u in event.mimeData().urls():
                if Path(u.toLocalFile()).suffix.lower() in ALL_EXTS:
                    self._hover = True
                    self.update()
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._hover = False
        self.update()

    def dropEvent(self, event):
        self._hover = False
        self.update()
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if Path(p).suffix.lower() in ALL_EXTS:
                self.file_dropped.emit(p)
                return

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QColor(C_AMBER)
        bg.setAlpha(16 if self._hover else 0)
        p.setBrush(bg)
        border = QColor(C_AMBER if self._hover else C_BORDER)
        pen = QPen(border, 1.4, Qt.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 12, 12)

        # 小图标：三根声柱
        cx, cy = int(36 * FONT_SCALE), h / 2
        p.setPen(Qt.NoPen)
        bar_brush = QBrush(QColor(C_AMBER if self._hover else C_WARN))
        p.setBrush(bar_brush)
        for dx, bh in ((-10, 12), (0, 22), (10, 16)):
            p.drawRoundedRect(
                QRectF(cx + dx * FONT_SCALE - 2.5 * FONT_SCALE, cy - bh * FONT_SCALE / 2,
                       5 * FONT_SCALE, bh * FONT_SCALE),
                2 * FONT_SCALE, 2 * FONT_SCALE)

        p.setPen(QPen(QColor(C_TEXT)))
        p.setFont(QFont("Microsoft YaHei UI", int(11 * FONT_SCALE), QFont.DemiBold))
        p.drawText(QPointF(int(64 * FONT_SCALE), cy - int(4 * FONT_SCALE)),
                   "点击选择，或将音 / 视频文件拖入此处")
        p.setPen(QPen(QColor(C_DIM)))
        p.setFont(QFont("Microsoft YaHei UI", int(8.5 * FONT_SCALE)))
        p.drawText(QPointF(int(64 * FONT_SCALE), cy + int(16 * FONT_SCALE)),
                   "WAV / MP3 / FLAC / M4A / AAC / OGG · MP4 / MKV / MOV / WEBM …")
        p.end()


# ---------------------------------------------------------------- 主窗口

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("声鉴 · 语音深度伪造检测")
        scr = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(
            min(int(880 * FONT_SCALE), scr.width() - 40),
            min(int(620 * FONT_SCALE), scr.height() - 40),
        )
        self._build_ui()
        self._center_on_screen()

    # ---- UI 构建

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 14)
        outer.setSpacing(14)

        # 顶栏
        header = QHBoxLayout()
        header.setSpacing(14)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("声鉴")
        title.setObjectName("Title")
        subtitle = QLabel("Spectra-AASIST3 · 语音深度伪造检测工作台")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)

        chip_model = QLabel("模型  Spectra-AASIST3 · INT8")
        chip_model.setObjectName("Chip")
        chip_ram = QLabel()
        chip_ram.setObjectName("Chip")
        chip_ram.setProperty("accent", "1")
        header.addWidget(chip_model)
        header.addWidget(chip_ram)
        outer.addLayout(header)

        # 提示行
        hint = QLabel(
            "提示：长音频仅前约 4 秒参与本次判定（图中高亮区）。视频将自动分离音轨；结果仅供辅助参考。"
        )
        hint.setObjectName("Hint")
        outer.addWidget(hint)

        # 主体分栏
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)

        # 左：上传 + 波形
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(12)

        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.on_file_selected)
        lv.addWidget(self.drop_zone)

        wave_card = QFrame()
        wave_card.setObjectName("Card")
        wv = QVBoxLayout(wave_card)
        wv.setContentsMargins(12, 12, 12, 12)
        wv.setSpacing(8)
        wave_head = QHBoxLayout()
        self.wave_title = QLabel("波形 · 原始 PCM")
        self.wave_title.setObjectName("CardTitle")
        self.wave_meta = QLabel("等待载入…")
        self.wave_meta.setObjectName("Dim")
        wave_head.addWidget(self.wave_title)
        wave_head.addStretch(1)
        wave_head.addWidget(self.wave_meta)
        wv.addLayout(wave_head)
        self.waveform = WaveformWidget()
        wv.addWidget(self.waveform, 1)
        lv.addWidget(wave_card, 1)

        # 操作行
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.btn_open = QPushButton("选择文件")
        self.btn_open.setObjectName("Ghost")
        self.btn_detect = QPushButton("开始检测")
        self.btn_detect.setObjectName("Primary")
        self.btn_detect.setEnabled(False)
        self.status_label = QLabel("待命")
        self.status_label.setObjectName("Dim")
        self.progress = QProgressBar()
        self.progress.setObjectName("Busy")
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(int(4 * FONT_SCALE))
        self.progress.setTextVisible(False)
        self.progress.hide()
        actions.addWidget(self.btn_open)
        actions.addWidget(self.btn_detect)
        actions.addSpacing(8)
        actions.addWidget(self.status_label, 1)
        actions.addWidget(self.progress)
        lv.addLayout(actions)

        # 右：结果
        right = QFrame()
        right.setObjectName("Card")
        right.setMinimumWidth(int(280 * FONT_SCALE))
        rv = QVBoxLayout(right)
        rv.setContentsMargins(18, 16, 18, 16)
        rv.setSpacing(12)

        r_head = QLabel("判定结果")
        r_head.setObjectName("CardTitle")
        rv.addWidget(r_head)

        self.verdict_label = QLabel("—")
        self.verdict_label.setObjectName("Verdict")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.verdict_label)

        self.conf_label = QLabel("置信度  —")
        self.conf_label.setObjectName("Conf")
        self.conf_label.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.conf_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setObjectName("ConfBar")
        self.conf_bar.setRange(0, 1000)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(int(10 * FONT_SCALE))
        rv.addWidget(self.conf_bar)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.prob_spoof_v = QLabel("—")
        self.prob_bona_v = QLabel("—")
        for row, (name, val, color) in enumerate((
            ("P · 伪造 spoof", self.prob_spoof_v, C_SPOOF),
            ("P · 真实 bona fide", self.prob_bona_v, C_BONA),
        )):
            lab = QLabel(name)
            lab.setObjectName("Dim")
            val.setObjectName("Metric")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setStyleSheet(f"color: {color};")
            grid.addWidget(lab, row, 0)
            grid.addWidget(val, row, 1)
        rv.addLayout(grid)

        self.time_v = QLabel("—")
        self.time_v.setObjectName("Metric")
        self.load_note = QLabel("")
        self.load_note.setObjectName("Faint")
        row_t = QLabel("推理耗时")
        row_t.setObjectName("Dim")
        tl = QHBoxLayout()
        tl.addWidget(row_t)
        tl.addStretch(1)
        tl.addWidget(self.time_v)
        rv.addLayout(tl)
        rv.addWidget(self.load_note)

        sep = QFrame()
        sep.setObjectName("Sep")
        sep.setFrameShape(QFrame.HLine)
        rv.addWidget(sep)

        file_card = QLabel("尚未选择文件")
        file_card.setObjectName("Dim")
        file_card.setWordWrap(True)
        rv.addWidget(file_card)
        self.file_card = file_card

        rv.addStretch(1)

        disclaimer = QLabel("检测结果由机器学习模型给出，可能出错，不构成司法鉴定依据。")
        disclaimer.setObjectName("Faint")
        disclaimer.setWordWrap(True)
        rv.addWidget(disclaimer)

        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        outer.addWidget(self.splitter, 1)

        # 底部
        footer = QLabel("本地推理 · 模型首次使用时自动加载 · 音频不上传")
        footer.setObjectName("Faint")
        outer.addWidget(footer)

        # RAM 建议
        avail_mb = get_available_ram_mb()
        lo, hi = recommend_duration(avail_mb)
        chip_ram.setText(f"可用内存 {avail_mb / 1024:.1f} GB · 建议样例 {lo}–{hi} s")
        chip_ram.setToolTip(
            f"按当前可用内存 {avail_mb / 1024:.2f} GB 估算：预留约 1 GB 给模型与界面后，"
            f"建议上传不超过 {hi} 秒的样例以保证解码与绘制流畅。"
        )

        # 信号
        self.btn_open.clicked.connect(lambda: self.drop_zone._pick_file())
        self.btn_detect.clicked.connect(self.start_detect)
        self._current_path = None
        self._vertical = False

    # ---- 自适应

    def resizeEvent(self, event):
        super().resizeEvent(event)
        vertical = event.size().width() < 1600
        if vertical != self._vertical:
            self._vertical = vertical
            self.splitter.setOrientation(Qt.Vertical if vertical else Qt.Horizontal)

    def _center_on_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(int(screen.width() * 0.88), int(1680 * FONT_SCALE))
        h = min(int(screen.height() * 0.88), int(1050 * FONT_SCALE))
        w = min(max(w, 640), screen.width() - 20)
        h = min(max(h, 480), screen.height() - 20)
        self.resize(w, h)
        self.move(
            screen.x() + (screen.width() - w) // 2,
            screen.y() + (screen.height() - h) // 2,
        )

    # ---- 流程

    def on_file_selected(self, path: str):
        self._current_path = path
        p = Path(path)
        self.file_card.setText(f"当前文件\n{p.name}")
        self.file_card.setToolTip(path)
        self.btn_detect.setEnabled(True)
        self._reset_result()
        self.start_detect()

    def _reset_result(self):
        self.verdict_label.setText("—")
        self.verdict_label.setProperty("tone", "none")
        self.verdict_label.style().unpolish(self.verdict_label)
        self.verdict_label.style().polish(self.verdict_label)
        self.conf_label.setText("置信度  —")
        self.conf_bar.setValue(0)
        self.prob_spoof_v.setText("—")
        self.prob_bona_v.setText("—")
        self.time_v.setText("—")
        self.load_note.setText("")

    def start_detect(self):
        if not self._current_path:
            return
        self.btn_detect.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.show()
        self.status_label.setText("启动中…")
        self.worker = DetectWorker(self._current_path, self)
        self.worker.stage.connect(self.status_label.setText)
        self.worker.finished_ok.connect(self.on_result)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_result(self, r: dict):
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self.btn_detect.setEnabled(True)

        audio = r["audio"]
        self.waveform.set_audio(audio)
        dur = audio.shape[0] / SAMPLE_RATE
        used = min(dur, MODEL_WINDOW_S)
        self.wave_meta.setText(f"全长 {dur:.2f}s · 本次分析前 {used:.2f}s")
        self.status_label.setText("完成")

        media = r["media"]
        bits = []
        if media.get("format_name"):
            bits.append(media["format_name"].split(",")[0])
        if media.get("duration"):
            bits.append(f"{media['duration']:.1f}s 容器时长")
        bits.append(f"解码 {audio.shape[0] / SAMPLE_RATE:.2f}s @ 16kHz")
        self.file_card.setText(
            f"{Path(self._current_path).name}\n" + " · ".join(bits)
        )

        bona = r["verdict"] == "bonafide"
        self.verdict_label.setText("真实 BONAFIDE" if bona else "伪造 SPOOF")
        tone = "bona" if bona else "spoof"
        self.verdict_label.setProperty("tone", tone)
        self.verdict_label.style().unpolish(self.verdict_label)
        self.verdict_label.style().polish(self.verdict_label)

        conf = r["confidence"]
        self.conf_label.setText(f"置信度  {conf * 100:.1f}%")
        self.conf_bar.setValue(int(conf * 1000))
        self.conf_bar.setProperty("tone", tone)
        self.conf_bar.style().unpolish(self.conf_bar)
        self.conf_bar.style().polish(self.conf_bar)

        self.prob_spoof_v.setText(f"{r['p_spoof'] * 100:.1f}%")
        self.prob_bona_v.setText(f"{r['p_bona'] * 100:.1f}%")
        self.time_v.setText(f"{r['infer_ms']:.1f} ms")
        if r["first_load"]:
            self.load_note.setText(f"首次加载模型 {r['load_ms'] / 1000:.1f}s（已缓存，后续跳过）")

    def on_error(self, msg: str):
        self.progress.hide()
        self.btn_open.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.status_label.setText("失败")
        QMessageBox.warning(self, "检测失败", msg)


# ---------------------------------------------------------------- 样式表

QSS = f"""
QWidget#Root {{
    background: {C_BG};
}}
QFrame#Card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 14px;
}}
QLabel {{
    color: {C_TEXT};
    background: transparent;
}}
QLabel#Title {{
    font-size: {24 * FONT_SCALE}px;
    font-weight: 700;
    letter-spacing: 6px;
    color: {C_TEXT};
    padding-left: 4px;
}}
QLabel#Subtitle {{
    font-size: {int(10.5 * FONT_SCALE)}px;
    color: {C_DIM};
    letter-spacing: 1px;
    padding-left: 6px;
}}
QLabel#Chip {{
    background: {C_CARD2};
    border: 1px solid {C_BORDER};
    border-radius: {9 * FONT_SCALE}px;
    color: {C_DIM};
    font-size: {10 * FONT_SCALE}px;
    padding: {6 * FONT_SCALE}px {12 * FONT_SCALE}px;
}}
QLabel#Chip[accent="1"] {{
    color: {C_AMBER};
    border-color: {C_BORDER};
}}
QLabel#Hint {{
    color: {C_WARN};
    font-size: {int(10.5 * FONT_SCALE)}px;
    padding: {2 * FONT_SCALE}px {6 * FONT_SCALE}px;
}}
QLabel#CardTitle {{
    font-size: {11 * FONT_SCALE}px;
    font-weight: 600;
    letter-spacing: 3px;
    color: {C_DIM};
}}
QLabel#Dim {{ color: {C_DIM}; font-size: {int(10.5 * FONT_SCALE)}px; }}
QLabel#Faint {{ color: {C_FAINT}; font-size: {int(9.5 * FONT_SCALE)}px; }}
QLabel#Metric {{
    font-size: {16 * FONT_SCALE}px;
    font-weight: 600;
    font-family: "Cascadia Mono", "Consolas", monospace;
    color: {C_TEXT};
}}
QLabel#Verdict {{
    font-size: {26 * FONT_SCALE}px;
    font-weight: 700;
    letter-spacing: 4px;
    padding: {14 * FONT_SCALE}px 0 {4 * FONT_SCALE}px 0;
    color: {C_FAINT};
}}
QLabel#Verdict[tone="bona"] {{ color: {C_BONA}; }}
QLabel#Verdict[tone="spoof"] {{ color: {C_SPOOF}; }}
QLabel#Conf {{
    font-size: {12 * FONT_SCALE}px;
    color: {C_DIM};
    padding-bottom: {4 * FONT_SCALE}px;
}}
QFrame#Sep {{
    border: none;
    border-top: 1px solid {C_BORDER_SOFT};
    max-height: 1px;
}}
QPushButton {{
    border-radius: {10 * FONT_SCALE}px;
    padding: {9 * FONT_SCALE}px {22 * FONT_SCALE}px;
    font-size: {12 * FONT_SCALE}px;
    font-weight: 600;
    letter-spacing: 2px;
    outline: none;
}}
QPushButton#Primary {{
    background: {C_AMBER};
    color: {C_ON_AMBER};
    border: none;
}}
QPushButton#Primary:hover {{ background: {C_AMBER_HI}; }}
QPushButton#Primary:pressed {{ background: #c88e42; }}
QPushButton#Primary:disabled {{
    background: #4a3d2c;
    color: #8a7a62;
}}
QPushButton#Ghost {{
    background: transparent;
    color: {C_WARN};
    border: 1px solid {C_BORDER};
}}
QPushButton#Ghost:hover {{
    border-color: {C_AMBER};
    color: {C_AMBER};
}}
QPushButton#Ghost:disabled {{ color: {C_FAINT}; }}
QProgressBar#Busy {{
    background: {C_BORDER_SOFT};
    border: none;
    border-radius: 2px;
}}
QProgressBar#Busy::chunk {{
    background: {C_AMBER};
    border-radius: 2px;
}}
QProgressBar#ConfBar {{
    background: {C_CARD2};
    border: 1px solid {C_BORDER_SOFT};
    border-radius: 5px;
}}
QProgressBar#ConfBar::chunk {{
    background: {C_FAINT};
    border-radius: 4px;
}}
QProgressBar#ConfBar[tone="bona"]::chunk {{ background: {C_BONA}; }}
QProgressBar#ConfBar[tone="spoof"]::chunk {{ background: {C_SPOOF}; }}
QSplitter::handle {{
    background: transparent;
    width: 6px;
    height: 6px;
}}
QScrollBar:vertical {{
    background: {C_BG};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""


def make_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(C_AMBER))
    p.drawEllipse(QRectF(2, 2, 60, 60))
    p.setBrush(QColor(C_ON_AMBER))
    for dx, bh in ((-14, 16), (-4, 30), (6, 22), (16, 12)):
        p.drawRoundedRect(QRectF(32 + dx - 3, 32 - bh / 2, 6, bh), 3, 3)
    p.end()
    return QIcon(pm)


def main():
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Microsoft YaHei UI", int(10 * FONT_SCALE)))
    app.setWindowIcon(make_icon())
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
