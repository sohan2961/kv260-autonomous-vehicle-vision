#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
from collections import defaultdict

TEST_CLASSES = [0, 1, 2, 3, 5, 7]
CLASS_NAMES = {0:"person",1:"bicycle",2:"car",3:"motorbike",5:"bus",7:"truck"}

GT_FILE = "carla_ground_truth_run_01.csv"
GT_FRAMES_FILE = "carla_frames_run_01.csv"
PRED_FILE = "kv260_predictions_run_01.csv"
PRED_FRAMES_FILE = "kv260_prediction_frames_run_01.csv"

OUT_SUMMARY = "kv260_accuracy_summary_run_01.txt"
OUT_CLASS = "kv260_accuracy_per_class_run_01.csv"

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    area_a = max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    area_b = max(0.0,bx2-bx1)*max(0.0,by2-by1)
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0

def parse_manifest(rows, key):
    out = []
    for r in rows:
        try: out.append(int(r[key]))
        except Exception: pass
    return out

def load_gt(rows, max_frame):
    gt = defaultdict(list)
    counts = defaultdict(int)
    for r in rows:
        try:
            frame = int(r["sent_frame_id"])
            cls = int(r["class_id"])
            if not (1 <= frame <= max_frame) or cls not in TEST_CLASSES:
                continue
            box = tuple(float(r[k]) for k in ("x1","y1","x2","y2"))
        except Exception:
            continue
        gt[(cls,frame)].append(box)
        counts[cls] += 1
    return gt, counts

def load_preds(rows, max_frame):
    preds = defaultdict(list)
    counts = defaultdict(int)
    for r in rows:
        try:
            frame = int(r["frame_id"])
            cls = int(r["class_id"])
            if not (1 <= frame <= max_frame) or cls not in TEST_CLASSES:
                continue
            score = float(r["confidence"])
            box = tuple(float(r[k]) for k in ("x1","y1","x2","y2"))
        except Exception:
            continue
        preds[cls].append((score, frame, box))
        counts[cls] += 1
    for cls in TEST_CLASSES:
        preds[cls].sort(key=lambda x: x[0], reverse=True)
    return preds, counts

def match_class(cls, pred_list, gt_by_key, iou_thr):
    class_frames = {}
    matched = {}
    gt_count = 0
    for (gt_cls, frame), boxes in gt_by_key.items():
        if gt_cls == cls:
            class_frames[frame] = boxes
            matched[frame] = [False]*len(boxes)
            gt_count += len(boxes)

    outcomes = []
    for score, frame, pbox in pred_list:
        boxes = class_frames.get(frame, [])
        flags = matched.get(frame, [])
        best_iou, best_idx = -1.0, -1
        for idx, gbox in enumerate(boxes):
            if flags[idx]:
                continue
            ov = iou(pbox, gbox)
            if ov > best_iou:
                best_iou, best_idx = ov, idx
        if best_idx >= 0 and best_iou >= iou_thr:
            flags[best_idx] = True
            outcomes.append((score,1))
        else:
            outcomes.append((score,0))
    return outcomes, gt_count

def ap101(outcomes, gt_count):
    if gt_count <= 0:
        return None
    tp = fp = 0
    recalls, precisions = [], []
    for _, is_tp in outcomes:
        if is_tp: tp += 1
        else: fp += 1
        precisions.append(tp/float(tp+fp))
        recalls.append(tp/float(gt_count))
    total = 0.0
    for i in range(101):
        r = i/100.0
        best = 0.0
        for rec, prec in zip(recalls, precisions):
            if rec >= r and prec > best:
                best = prec
        total += best
    return total/101.0

def metrics_at_threshold(preds, gt, conf_thr, iou_thr=0.5):
    TPs=FPs=FNs=0
    per={}
    for cls in TEST_CLASSES:
        selected=[p for p in preds[cls] if p[0] >= conf_thr]
        outcomes, gt_count = match_class(cls, selected, gt, iou_thr)
        tp=sum(v for _,v in outcomes)
        fp=len(outcomes)-tp
        fn=max(0,gt_count-tp)
        p=tp/float(tp+fp) if tp+fp else 0.0
        r=tp/float(tp+fn) if tp+fn else 0.0
        f1=2*p*r/(p+r) if p+r else 0.0
        per[cls]=(tp,fp,fn,p,r,f1,len(selected),gt_count)
        TPs += tp; FPs += fp; FNs += fn
    p=TPs/float(TPs+FPs) if TPs+FPs else 0.0
    r=TPs/float(TPs+FNs) if TPs+FNs else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0
    return TPs,FPs,FNs,p,r,f1,per

def main():
    required=[GT_FILE,GT_FRAMES_FILE,PRED_FILE,PRED_FRAMES_FILE]
    missing=[x for x in required if not os.path.isfile(x)]
    if missing:
        print("ERROR: missing files:")
        for x in missing: print("  "+x)
        print("Run this script from the Windows results folder.")
        return 1

    gt_frames=read_csv(GT_FRAMES_FILE)
    pred_frames=read_csv(PRED_FRAMES_FILE)
    gt_rows=read_csv(GT_FILE)
    pred_rows=read_csv(PRED_FILE)

    carla_ids=parse_manifest(gt_frames,"sent_frame_id")
    kv_ids=parse_manifest(pred_frames,"frame_id")
    max_common=min(max(carla_ids),max(kv_ids))

    gt,gt_counts=load_gt(gt_rows,max_common)
    preds,pred_counts=load_preds(pred_rows,max_common)

    thresholds=[round(0.50+0.05*i,2) for i in range(10)]
    ap={}
    for cls in TEST_CLASSES:
        ap[cls]={}
        for t in thresholds:
            outcomes,n=match_class(cls,preds[cls],gt,t)
            ap[cls][t]=ap101(outcomes,n)

    present=[c for c in TEST_CLASSES if gt_counts.get(c,0)>0]
    map50=sum(ap[c][0.50] for c in present)/len(present) if present else 0.0
    perclass_5095={}
    for c in TEST_CLASSES:
        vals=[ap[c][t] for t in thresholds if ap[c][t] is not None]
        perclass_5095[c]=sum(vals)/len(vals) if vals else None
    map5095=sum(perclass_5095[c] for c in present)/len(present) if present else 0.0

    m050=metrics_at_threshold(preds,gt,0.50,0.50)

    # Grid-search practical best F1 threshold.
    cands=[0.001+i*(0.099/99.0) for i in range(100)]
    cands += [0.10+i*(0.85/170.0) for i in range(171)]
    cands += [0.50]
    best=None
    for t in sorted(set(round(x,6) for x in cands)):
        m=metrics_at_threshold(preds,gt,t,0.50)
        if best is None or m[5] > best[1][5]:
            best=(t,m)

    def pct(x): return "{:.2f}%".format(100*x)

    lines=[]
    lines.append("KV260 CARLA OBJECT-DETECTION ACCURACY - RUN 01")
    lines.append("="*58)
    lines.append("Evaluation frames: 1..{}".format(max_common))
    lines.append("CARLA max sent_frame_id: {}".format(max(carla_ids)))
    lines.append("KV260 max frame_id: {}".format(max(kv_ids)))
    lines.append("Ground-truth objects in common range: {}".format(sum(gt_counts.values())))
    lines.append("Logged predictions in common range: {}".format(sum(pred_counts.values())))
    lines.append("GT-present classes: {}".format(", ".join(CLASS_NAMES[c] for c in present) if present else "none"))
    lines.append("")
    lines.append("mAP@0.50      = {}".format(pct(map50)))
    lines.append("mAP@0.50:0.95 = {}".format(pct(map5095)))
    lines.append("")
    lines.append("At confidence >= 0.50 and IoU >= 0.50")
    lines.append("TP = {}  FP = {}  FN = {}".format(m050[0],m050[1],m050[2]))
    lines.append("Precision = {}".format(pct(m050[3])))
    lines.append("Recall    = {}".format(pct(m050[4])))
    lines.append("F1-score  = {}".format(pct(m050[5])))
    lines.append("")
    lines.append("Best pooled F1 at IoU >= 0.50")
    lines.append("Confidence threshold = {:.4f}".format(best[0]))
    lines.append("TP = {}  FP = {}  FN = {}".format(best[1][0],best[1][1],best[1][2]))
    lines.append("Precision = {}".format(pct(best[1][3])))
    lines.append("Recall    = {}".format(pct(best[1][4])))
    lines.append("F1-score  = {}".format(pct(best[1][5])))
    lines.append("")
    lines.append("Per-class AP")
    for c in TEST_CLASSES:
        a50=ap[c][0.50]
        a5095=perclass_5095[c]
        lines.append("{:<10s} GT={:<4d} AP50={} AP50:95={}".format(
            CLASS_NAMES[c], gt_counts.get(c,0),
            "N/A" if a50 is None else pct(a50),
            "N/A" if a5095 is None else pct(a5095)
        ))
    lines.append("")
    lines.append("NOTE: CARLA GT is projected from 3D actor boxes; occlusion masks are not modeled.")
    lines.append("This is therefore a CARLA projected-ground-truth benchmark, not official COCO evaluation.")

    summary="\n".join(lines)
    print(summary)

    with open(OUT_SUMMARY,"w",encoding="utf-8") as f:
        f.write(summary+"\n")

    with open(OUT_CLASS,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["class_id","class_name","gt_count","logged_prediction_count","AP50","AP50_95"])
        for c in TEST_CLASSES:
            w.writerow([
                c,CLASS_NAMES[c],gt_counts.get(c,0),pred_counts.get(c,0),
                "" if ap[c][0.50] is None else "{:.6f}".format(ap[c][0.50]),
                "" if perclass_5095[c] is None else "{:.6f}".format(perclass_5095[c])
            ])

    print("\nSaved:")
    print(os.path.abspath(OUT_SUMMARY))
    print(os.path.abspath(OUT_CLASS))
    return 0

if __name__=="__main__":
    sys.exit(main())
