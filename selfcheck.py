# -*- coding: utf-8 -*-
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PyQt5.QtWidgets import QApplication

import app as A


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(A.QSS)
    w = A.MainWindow()
    w.show()
    w._current_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.wav")
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, dtype=np.float32)
    audio = (0.4 * np.sin(2 * np.pi * 220 * t) * np.exp(-t / 2)).astype(np.float32)
    w.waveform.set_audio(audio)
    w.on_result({
        "audio": audio,
        "media": {"format_name": "wav", "duration": 3.0, "bit_rate": ""},
        "p_spoof": 0.12, "p_bona": 0.88,
        "verdict": "bonafide", "confidence": 0.88,
        "infer_ms": 142.6, "load_ms": 6500.0, "first_load": True,
    })
    app.processEvents()
    print("UI self-check OK; size = %dx%d" % (w.width(), w.height()))


if __name__ == "__main__":
    main()
