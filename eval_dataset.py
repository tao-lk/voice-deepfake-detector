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


def best_threshold(scores: np.ndarray, labels: np.ndarray):
    """扫描所有候选阈值，返回多个准则下的最优阈值及对应指标。

    返回 dict: key -> (threshold, accuracy, f1, fpr, fnr)
    准则: eer(FNR=FPR工作点), acc(最高准确率), f1(最高F1), youden(最大J=TPR-FPR)
    """
    n_pos = int(labels.sum())
    n_neg = int(labels.shape[0] - n_pos)
    cands = np.unique(scores)
    rows = []
    for t in cands:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        tn = int((~pred & (labels == 0)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        acc = (tp + tn) / labels.shape[0]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        fpr = fp / n_neg
        fnr = fn / n_pos
        rows.append((float(t), acc, f1, fpr, fnr))

    arr = np.array(rows)
    ts, accs, f1s, fprs, fnrs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    def pack(idx):
        i = int(idx)
        return (float(ts[i]), float(accs[i]), float(f1s[i]), float(fprs[i]), float(fnrs[i]))

    out = {
        "acc": pack(np.argmax(accs)),
        "f1": pack(np.argmax(f1s)),
        "youden": pack(np.argmax((1 - fprs) - fnrs)),
    }
    # EER 工作点: |FNR - FPR| 最小处
    out["eer"] = pack(np.argmin(np.abs(fnrs - fprs)))
    return out


def collect_files(dataset_dir: Path, real_dir: str = "real", spoof_dirs: str = "fake"):
    """收集样本。real_dir 为真实音频子目录，spoof_dirs 为伪造子目录（逗号分隔）。

    返回 items: (path, label, group)，group 为子目录名，用于分攻击类型统计。
    """
    items = []
    real_path = dataset_dir / real_dir
    if not real_path.is_dir():
        raise FileNotFoundError(real_path)
    for p in sorted(real_path.glob("*.wav")):
        items.append((p, 1, real_dir))
    for sub in spoof_dirs.split(","):
        sub = sub.strip()
        d = dataset_dir / sub
        if not d.is_dir():
            raise FileNotFoundError(d)
        for p in sorted(d.glob("*.wav")):
            items.append((p, 0, sub))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=r"E:\pytorch深度学习\信号处理\语音伪造检测mfcc+cnn+attention\dataset\train",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="每类最多取 N 条，0=全量")
    parser.add_argument("--real-dir", default="real", help="真实音频子目录名")
    parser.add_argument("--spoof-dirs", default="fake", help="伪造音频子目录名，逗号分隔")
    parser.add_argument("--out-prefix", default="eval", help="输出文件前缀")
    args = parser.parse_args()

    logger.add(f"{args.out_prefix}_scores.log", rotation="10 MB", encoding="utf-8")
    logger.info("加载模型: {}", A.MODEL_PATH.name)
    sess, load_ms = A.get_session()
    logger.info("模型加载完成 {:.1f} ms", load_ms)

    items = collect_files(Path(args.dataset), args.real_dir, args.spoof_dirs)
    if args.limit > 0:
        kept, per_group = [], {}
        for it in items:
            g = it[2]
            if per_group.get(g, 0) < args.limit:
                kept.append(it)
                per_group[g] = per_group.get(g, 0) + 1
        items = kept
    logger.info("数据集: {} 条 (real={} + spoof={}), batch={}", len(items), args.real_dir, args.spoof_dirs, args.batch)

    scores, labels, groups = [], [], []
    errors = 0
    t_start = time.perf_counter()

    for i in tqdm(range(0, len(items), args.batch), desc="推理", unit="batch"):
        chunk = items[i : i + args.batch]
        batch_x = []
        batch_y = []
        batch_g = []
        for path, label, group in chunk:
            try:
                audio = load_audio_16k(path)
                batch_x.append(A.make_model_input(audio))
                batch_y.append(label)
                batch_g.append(group)
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
        groups.extend(batch_g)

    elapsed = time.perf_counter() - t_start
    scores = np.array(scores)
    labels = np.array(labels, dtype=np.int64)
    groups = np.array(groups)

    def metrics_at(scores, labels, thr):
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        tn = int((~pred & (labels == 0)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        acc = (tp + tn) / max(labels.shape[0], 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return tp, tn, fp, fn, acc, prec, rec, f1

    n_real = int(labels.sum())
    n_fake = int(labels.shape[0] - n_real)
    tp, tn, fp, fn, acc, precision, recall, f1 = metrics_at(scores, labels, 0.5)
    eer = compute_eer(scores, labels)
    auc = compute_auc(scores, labels)
    cm05 = (tp, tn, fp, fn)

    logger.info("==== 评估结果 @0.5 ====")
    logger.info("样本: {} (real={}, fake={}), 失败 {}", labels.shape[0], n_real, n_fake, errors)
    logger.info("Accuracy : {:.2f}%  (TP={} TN={} FP={} FN={})", acc * 100, tp, tn, fp, fn)
    logger.info("Precision: {:.2f}%   Recall: {:.2f}%   F1: {:.4f}", precision * 100, recall * 100, f1)
    logger.info("EER      : {:.2f}%", eer * 100)
    logger.info("AUC      : {:.4f}", auc)

    out_npz = Path(f"{args.out_prefix}_scores.npz")
    np.savez(out_npz, scores=scores, labels=labels, groups=groups)
    logger.info("概率已保存: {}", out_npz.resolve())

    bests = best_threshold(scores, labels)
    logger.info("==== 最优阈值扫描 ====")
    for name, key in (("EER工作点", "eer"), ("最高准确率", "acc"), ("最高F1", "f1"), ("最大Youden J", "youden")):
        t, a, f, fpr, fnr = bests[key]
        logger.info(
            "{:<10} threshold={:.10f}  Acc={:.2f}%  F1={:.4f}  FPR={:.2f}%  FNR={:.2f}%",
            name, t, a * 100, f, fpr * 100, fnr * 100,
        )

    t_ui = A.THRESHOLD_BONA
    tp, tn, fp, fn, acc_ui, _, _, f1_ui = metrics_at(scores, labels, t_ui)
    logger.info("==== UI 阈值 {:.10f} ====", t_ui)
    logger.info("Accuracy : {:.2f}%  F1: {:.4f}  (TP={} TN={} FP={} FN={})", acc_ui * 100, f1_ui, tp, tn, fp, fn)

    logger.info("==== 分攻击类型 @UI阈值 ====")
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        g_scores, g_labels = scores[mask], labels[mask]
        tp, tn, fp, fn, g_acc, _, _, _ = metrics_at(g_scores, g_labels, t_ui)
        if g_labels.min() == g_labels.max():
            # 单类别子集: F1/EER/AUC 无定义, 只报 Acc 与错误数
            role = "真实" if g_labels[0] == 1 else "伪造"
            logger.info("{:<6} n={:<5} {} Acc={:.2f}%  误判数={}", g, int(mask.sum()), role, g_acc * 100, fp + fn)
        else:
            g_eer = compute_eer(g_scores, g_labels)
            g_auc = compute_auc(g_scores, g_labels)
            logger.info(
                "{:<6} n={:<5} Acc={:.2f}%  EER={:.2f}%  AUC={:.4f}  FP={} FN={}",
                g, int(mask.sum()), g_acc * 100, g_eer * 100, g_auc, fp, fn,
            )
    logger.info("平均耗时 : {:.0f} ms/条 (含IO, batch={})", elapsed / labels.shape[0] * 1000, args.batch)

    print()
    print("=" * 52)
    print(f"  样本数     : {labels.shape[0]}  (real={n_real}, spoof={n_fake})")
    print(f"  --- 阈值 0.5 ---")
    print(f"  Accuracy   : {acc * 100:6.2f} %")
    print(f"  Precision  : {precision * 100:6.2f} %")
    print(f"  Recall     : {recall * 100:6.2f} %")
    print(f"  F1         : {f1:6.4f}")
    print(f"  EER        : {eer * 100:6.2f} %")
    print(f"  AUC        : {auc:6.4f}")
    print(f"  混淆矩阵   : TP={cm05[0]} TN={cm05[1]} FP={cm05[2]} FN={cm05[3]}")
    print(f"  总耗时     : {elapsed:.1f} s")
    print("-" * 52)
    print("  最优阈值扫描 (基于 P(real)=p_bona):")
    for name, key in (("EER工作点", "eer"), ("最高准确率", "acc"), ("最高F1", "f1"), ("最大Youden J", "youden")):
        t, a, f, fpr, fnr = bests[key]
        print(f"  {name:<8}: threshold={t:.10f}  Acc={a * 100:6.2f}%  F1={f:.4f}  FPR={fpr * 100:5.2f}%  FNR={fnr * 100:5.2f}%")
    tp, tn, fp, fn, acc_ui, prec_ui, rec_ui, f1_ui = metrics_at(scores, labels, t_ui)
    print(f"  --- UI 阈值 {t_ui:.10f} ---")
    print(f"  Accuracy   : {acc_ui * 100:6.2f} %   F1={f1_ui:.4f}   FP={fp} FN={fn}")
    print("  分攻击类型 @UI阈值:")
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        g_scores, g_labels = scores[mask], labels[mask]
        tp, tn, fp, fn, g_acc, _, _, _ = metrics_at(g_scores, g_labels, t_ui)
        if g_labels.min() == g_labels.max():
            role = "real" if g_labels[0] == 1 else "spoof"
            print(f"    {g:<6} n={mask.sum():<5} {role:<5} Acc={g_acc * 100:6.2f}%  误判={fp + fn}")
        else:
            g_eer = compute_eer(g_scores, g_labels)
            g_auc = compute_auc(g_scores, g_labels)
            print(f"    {g:<6} n={mask.sum():<5} Acc={g_acc * 100:6.2f}%  EER={g_eer * 100:5.2f}%  AUC={g_auc:.4f}  FP={fp} FN={fn}")
    print(f"  概率已保存 : {out_npz.resolve()}")
    print("=" * 52)


if __name__ == "__main__":
    sys.exit(main())
