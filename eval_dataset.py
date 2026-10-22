# -*- coding: utf-8 -*-
"""在本地标注数据集上批量评估 Spectra-AASIST3 INT8 模型性能。

数据集: dataset/train/{fake,real}/*.wav (16kHz)
指标: EER / AUC / Accuracy / Precision / Recall / F1 / 混淆矩阵
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

import app as A


def load_audio_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != A.SAMPLE_RATE:
        n_out = int(round(data.shape[0] * A.SAMPLE_RATE / sr))
        data = np.interp(
            np.linspace(0, data.shape[0] - 1, n_out),
            np.arange(data.shape[0]),
            data,
        ).astype(np.float32)
    return data


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """scores: 越高越像 bona fide; labels: 1=real, 0=fake。

    按阈值从高到低扫描 ROC，取 FNR 与 FPR 首次相交处线性插值得 EER。
    """
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    fprs = [0.0]
    fnrs = [1.0]
    for t in np.unique(scores)[::-1]:
        pred = scores >= t
        fprs.append(float(((pred) & (labels == 0)).sum()) / n_neg)
        fnrs.append(float((~pred & (labels == 1)).sum()) / n_pos)
    fprs = np.array(fprs)
    fnrs = np.array(fnrs)
    for i in range(1, fprs.shape[0]):
        if fnrs[i] <= fprs[i]:
            x0, x1 = fprs[i - 1], fprs[i]
            y0, y1 = fnrs[i - 1], fnrs[i]
            dx, dy = x1 - x0, y1 - y0
            denom = dx - dy
            p = 0.0 if abs(denom) < 1e-12 else (y0 - x0) / denom
            p = min(max(p, 0.0), 1.0)
            return float((x0 + p * dx + y0 + p * dy) / 2)
    return float(max(fprs[-1], fnrs[-1]))


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    n_pos = labels.sum()
    n_neg = labels.shape[0] - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def collect_files(dataset_dir: Path):
    items = []
    for label, sub in ((0, "fake"), (1, "real")):
        d = dataset_dir / sub
        if not d.is_dir():
            raise FileNotFoundError(d)
        for p in sorted(d.glob("*.wav")):
            items.append((p, label))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=r"E:\pytorch深度学习\信号处理\语音伪造检测mfcc+cnn+attention\dataset\train",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="每类最多取 N 条，0=全量")
    args = parser.parse_args()

    logger.add("eval_train.log", rotation="10 MB", encoding="utf-8")
    logger.info("加载模型: {}", A.MODEL_PATH.name)
    sess, load_ms = A.get_session()
    logger.info("模型加载完成 {:.1f} ms", load_ms)

    items = collect_files(Path(args.dataset))
    if args.limit > 0:
        fake = [x for x in items if x[1] == 0][: args.limit]
        real = [x for x in items if x[1] == 1][: args.limit]
        items = fake + real
    logger.info("数据集: {} 条 (fake+real), batch={}", len(items), args.batch)

    scores, labels = [], []
    errors = 0
    t_start = time.perf_counter()

    for i in tqdm(range(0, len(items), args.batch), desc="推理", unit="batch"):
        chunk = items[i : i + args.batch]
        batch_x = []
        batch_y = []
        for path, label in chunk:
            try:
                audio = load_audio_16k(path)
                batch_x.append(A.make_model_input(audio))
                batch_y.append(label)
            except Exception as exc:
                errors += 1
                logger.warning("跳过 {}: {}", path.name, exc)
        if not batch_x:
            continue
        x = np.stack(batch_x)
        logits = sess.run(None, {"wav": x})[0]
        probs = A.softmax_batch(logits.astype(np.float64))
        scores.extend(probs[:, 1].tolist())
        labels.extend(batch_y)

    elapsed = time.perf_counter() - t_start
    scores = np.array(scores)
    labels = np.array(labels, dtype=np.int64)

    n_real = int(labels.sum())
    n_fake = int(labels.shape[0] - n_real)
    pred = (scores >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / labels.shape[0]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    eer = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)

    logger.info("==== 评估结果 ====")
    logger.info("样本: {} (real={}, fake={}), 失败 {}", labels.shape[0], n_real, n_fake, errors)
    logger.info("Accuracy : {:.2f}%  (TP={} TN={} FP={} FN={})", acc * 100, tp, tn, fp, fn)
    logger.info("Precision: {:.2f}%   Recall: {:.2f}%   F1: {:.4f}", precision * 100, recall * 100, f1)
    logger.info("EER      : {:.2f}%", eer * 100)
    logger.info("AUC      : {:.4f}", auc)
    logger.info("平均耗时 : {:.0f} ms/条 (含IO, batch={})", elapsed / labels.shape[0] * 1000, args.batch)

    print()
    print("=" * 46)
    print(f"  样本数     : {labels.shape[0]}  (real={n_real}, fake={n_fake})")
    print(f"  Accuracy   : {acc * 100:6.2f} %")
    print(f"  Precision  : {precision * 100:6.2f} %")
    print(f"  Recall     : {recall * 100:6.2f} %")
    print(f"  F1         : {f1:6.4f}")
    print(f"  EER        : {eer * 100:6.2f} %")
    print(f"  AUC        : {auc:6.4f}")
    print(f"  混淆矩阵   : TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  总耗时     : {elapsed:.1f} s")
    print("=" * 46)


if __name__ == "__main__":
    sys.exit(main())
