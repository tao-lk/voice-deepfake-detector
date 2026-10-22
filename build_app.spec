# -*- mode: python ; coding: utf-8 -*-
"""声鉴 PyInstaller 打包配置: onedir + 无控制台 + 精简依赖。

产物: dist/声鉴/声鉴.exe (自包含模型与 ffmpeg)
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# onnxruntime 仅用 CPU EP, 过滤 DirectML 等无用动态库
ort_binaries = [
    (dest, src)
    for dest, src in collect_dynamic_libs("onnxruntime")
    if "directml" not in dest.lower() and "directml" not in src.lower()
]

# 排除未使用的大体积模块, 控制包体积
EXCLUDES = [
    # Qt 未用到的重型模块
    "PyQt5.QtWebEngine", "PyQt5.QtWebEngineCore", "PyQt5.QtWebEngineWidgets",
    "PyQt5.QtQuick", "PyQt5.QtQuickWidgets", "PyQt5.QtQml",
    "PyQt5.QtSql", "PyQt5.QtTest", "PyQt5.QtNetwork",
    "PyQt5.QtMultimedia", "PyQt5.QtMultimediaWidgets",
    "PyQt5.QtOpenGL", "PyQt5.QtXml", "PyQt5.QtHelp",
    "PyQt5.QtDesigner", "PyQt5.QtBluetooth", "PyQt5.QtNfc",
    "PyQt5.QtPositioning", "PyQt5.QtLocation", "PyQt5.QtSensors",
    "PyQt5.QtSerialPort", "PyQt5.QtWebChannel", "PyQt5.QtWebSockets",
    "PyQt5.QtChart", "PyQt5.QtDataVisualization", "PyQt5.QtPdf",
    # 常见误收
    "tkinter", "matplotlib", "PIL", "IPython", "notebook",
    "torch", "tensorflow", "paddle", "jax", "sklearn",
    "onnxruntime.training", "onnxruntime.quantization",
    "unittest", "pydoc", "doctest",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=ort_binaries,
    datas=[
        (r"models\spectra-aasist3-int8-audio-deepfake\spectra-aasist3-int8-dynamic.onnx",
         r"models\spectra-aasist3-int8-audio-deepfake"),
        (r"bin\ffmpeg.exe", "bin"),
        (r"bin\ffprobe.exe", "bin"),
    ],
    hiddenimports=["onnxruntime", "onnxruntime.capi", "onnxruntime.capi.onnxruntime_pybind11_state"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="声鉴",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="声鉴",
)
