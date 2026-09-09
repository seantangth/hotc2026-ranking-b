#!/usr/bin/env python3
"""Ranking B 交付定案器——把兩套管線的原始輸出合併成最終提交檔。

流程（順序即為已實測驗證的插入點，勿調換）：
  1. 對【兩套】輸出各自套用 GT 座標慣例校正（上/左各 +1px，clamp≥0），但保留每序列首幀 init
  2. 跨底座凍結救援 splice：主線凍結串 ≥K 且來源活著 ⇒ 借來源框——D070/v039／K=6
  3. 可選 RedNIR 品質頭 v056（僅 ``--qhead v056`` 明示啟用）。
     fresh Ranking B 預設 ``--qhead none``，因現有頭未通過 exact-pair grouped OOF。
  4. formal validator：對 sample_submission 做 exact-set 比對，fail-closed

用法（Ranking B 三天窗口內）：
  python3 track_t1.py --backend sam3     ... --out-dir out_main    # 主線（含 crop）
  python3 track_t1.py --backend samurai  ... --out-dir out_src     # 來源（含 crop，merge base = e02）
  ⚠️ 兩套的 crop 窗【相同】（e15 ∪ e02），只有 merge 的 base 不同 ⇒ crop prep 只跑一次。
  ⚠️ 來源必須是 e23b 型（窗含 e15），**不是** e23a／v011——後者的窗需要 E03=DAM4SAM，
     而 track_t1.py 無此 backend ⇒ 用 v011 會讓交付物無法完整重現（D073）。
  python3 finalize_submission.py --main out_main/submission.csv \\
      --source out_src/submission.csv --sample sample_submisson.csv --out final.csv

自測（--selftest）：(a) 位元級重現 v049（校正＋splice K=6）；
(b) 再套 v056 頭，浮點對上 sub_v056_rn_iab03.csv（110 幀）；
(c) 位元級重現 v078＝現行單一窗交付鏈（LB 0.71666）；
(d) 位元級重現 v090＝crop 窗三方共識鏈（LB 0.71096，rankB_deliver_v090 profile）。
依 D067(f)「判別工具必須在已知答案的資料上自測」，**交付前必跑**。
"""
import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── 已定案的常數（改動任何一個都會改變交付結果，需新決策編號）────────────────
# D067：test GT 於上、左各外擴 1px（右/下對齊；1px 為精確峰值）
# ⚠️ 兩邊的【跨資料集信心不同】，Ranking B 可分別開關（見 --corr 選項與 D072）：
#   上緣：train GT（val 實測 +0.0028）與 test（+0.0091）**都有** ⇒ 跨資料集穩定，信心高
#   左緣：**只在 test 有**（+0.0051），train GT 左緣中位 0.0 ⇒ 信心低
CORR_TOP_PX = 1.0
CORR_LEFT_PX = 1.0
SPLICE_K = 6                # D070/D071：test 劑量曲線在 K∈[5,8] 為平台（0.71302/0.71309/0.71300），
                            # 依 D061 取平台中央 K=6（LB 亦最高 0.71309）；勿改回 5（那是平台下緣）
SPLICE_SRC_MAX_RUN = 2      # 來源凍結串 <2 才算「活著」
# v056（08-18 LB 0.71403）：RedNIR ∧ iab<0.30 ∧ 主線未進凍結救援 ∧ 來源活著 ∧ 頭選 B
QHEAD_IAB_MAX = 0.30
QHEAD_MARGIN = 0.0          # p = P(A 較好)；p < 0.5 - margin 才換 B
QHEAD_WEIGHTS = Path(__file__).resolve().parent / "hsot" / "qhead_weights_v056.npz"
# selector-v2（D085）：對 dIoU 做 gain-weighted ridge 迴歸，取代二元分類頭。
# 權重由 export_selector_weights.py 於 405 crop-merged 配對上一次 fit 後固化
# ——9/7 沒有 405 快取可重 fit（~6.5 A100-h），交付一律載入既有權重。
# τ 交付值 0.05：LB 劑量曲線 0.00/0.02/0.05/0.08/0.15 =
# 0.71532/0.71552/0.71550/0.71507/0.71457，平台 [0.00,0.05]（0.02 的 +0.00002 是噪聲）。
SELECTOR_WEIGHTS = Path(__file__).resolve().parent / "hsot" / "selector_weights_v2.npz"

# 🚨 selftest (c) 的凍結第三腿 fixture。**任何 run 都不得寫入這個目錄。**
# 它是 08-22 那次 rankB_robust test75 的產物，(c) 段靠它位元級重現 sub_v078。
# run_rankb_robust_test75_lambda_v1.sh 的 DEST 曾預設指向此處，跑一次對照檔演練
# 就會覆寫它、讓 9/7 主檔每次開工都在 (c) die（09-04 稽核 G09，已於該腳本加 fail-closed 擋下）。
FROZEN_THIRD_LEG_FIXTURE = ROOT / "5_outputs/rankb_robust_test75_20260822/run/full_sam3/submission.csv"

SUBMISSION_COLUMNS = ("ID", "x", "y", "width", "height")


class SubmissionValidationError(ValueError):
    """輸入 CSV 不符交付規格。不容許部分載入，避免 duplicate 被字典靜默覆寫。"""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _limited(errors, limit=8):
    errors = list(errors)
    return errors[:limit] + ([f"…另有 {len(errors) - limit} 筆"] if len(errors) > limit else [])


def load(p, label="submission", validate_boxes=True):
    """嚴格載入 submission CSV：ID 必須唯一，main/source 的 box 必須合法。

    官方 sample 的 box 是 0 placeholder，因此 ``validate_boxes=False`` 時只將它當作
    ID schema 與 canonical order，不將 placeholder 當成 prediction 驗證。
    """
    d = defaultdict(dict)
    order = []
    seen = set()
    errors = []
    p = Path(p)
    try:
        fh = p.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise SubmissionValidationError([f"{label}: 無法讀取 {p}: {exc}"]) from exc
    with fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        if fields is None:
            raise SubmissionValidationError([f"{label}: CSV 為空或缺 header"])
        missing_cols = [c for c in SUBMISSION_COLUMNS if c not in fields]
        if missing_cols:
            raise SubmissionValidationError([f"{label}: 缺欄位 {missing_cols}"])
        if len(fields) != len(set(fields)):
            raise SubmissionValidationError([f"{label}: header 有重複欄位"])

        for row_no, r in enumerate(reader, start=2):
            rid = (r.get("ID") or "").strip()
            prefix = f"{label} 第 {row_no} 列"
            if not rid:
                errors.append(f"{prefix}: ID 為空")
                continue
            if rid in seen:
                errors.append(f"{prefix}: 重複 ID {rid}")
                continue
            seen.add(rid)
            try:
                s, f_text = rid.rsplit("_", 1)
                f = int(f_text)
                if not s:
                    raise ValueError
            except ValueError:
                errors.append(f"{prefix}: ID 格式非 <sequence>_<integer-frame>: {rid}")
                continue
            if validate_boxes:
                try:
                    box = [float(r[c]) for c in SUBMISSION_COLUMNS[1:]]
                except (TypeError, ValueError):
                    errors.append(f"{prefix} {rid}: box 含非數值")
                    continue
                x, y, w, h = box
                if not all(math.isfinite(v) for v in box):
                    errors.append(f"{prefix} {rid}: box 含 NaN/Inf")
                    continue
                if x < 0 or y < 0:
                    errors.append(f"{prefix} {rid}: x/y 必須 >=0 ({x},{y})")
                    continue
                if w <= 0 or h <= 0:
                    errors.append(f"{prefix} {rid}: w/h 必須 >0 ({w},{h})")
                    continue
            else:
                box = [0.0, 0.0, 0.0, 0.0]
            if f in d[s]:
                errors.append(f"{prefix}: sequence/frame 衝突 {s}_{f}")
                continue
            d[s][f] = box
            order.append(rid)
    if not seen:
        errors.append(f"{label}: 沒有任何資料列")
    if errors:
        raise SubmissionValidationError(_limited(errors))
    return d, order


def apply_correction(d, top=CORR_TOP_PX, left=CORR_LEFT_PX, preserve_first=True):
    """上/左各外擴指定 px，並預設保留每序列最早幀的 raw init。

    ``preserve_first=False`` 僅供歷史 v049/v056 selftest 位元級重現舊行為。
    """
    out = defaultdict(dict)
    for s, fm in d.items():
        first = min(fm) if fm else None
        for f, (x, y, w, h) in fm.items():
            if preserve_first and f == first:
                out[s][f] = [x, y, w, h]
                continue
            nx = max(0.0, x - left)
            ny = max(0.0, y - top)
            out[s][f] = [nx, ny, w + (x - nx), h + (y - ny)]
    return out


def frozen_runs(fm):
    ks = sorted(fm)
    o, r = {}, 0
    for i, k in enumerate(ks):
        r = r + 1 if i > 0 and fm[k] == fm[ks[i - 1]] else 0
        o[k] = r
    return o


def splice(main, src, K=SPLICE_K):
    """主線凍結串 ≥K ∧ 來源活著 ⇒ 借來源框。因果、無 GT、不認序列名。"""
    RM = {s: frozen_runs(main[s]) for s in main}
    RS = {s: frozen_runs(src[s]) for s in src}
    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in main.items()}
    n = 0
    for s in main:
        if s not in src:
            continue
        for f in main[s]:
            if f in src[s] and RM[s][f] >= K and RS[s][f] < SPLICE_SRC_MAX_RUN:
                out[s][f] = list(src[s][f])
                n += 1
    return out, n


def apply_qhead_v056(spliced, main_corr, src_corr, weights_path=QHEAD_WEIGHTS,
                     iab_max=QHEAD_IAB_MAX, margin=QHEAD_MARGIN, K=SPLICE_K):
    """在 splice 之後、只動 RedNIR 不一致幀。特徵用校正後的主線／來源（不是 splice 後的框）。"""
    import numpy as np
    from hsot.quality_head_v1 import feats, frozen_runs, iou, modality, predict_p

    z = np.load(weights_path)
    w, mu, sd = z["w"], z["mu"], z["sd"]
    A = {s: {f: np.asarray(b, dtype=np.float64) for f, b in fm.items()} for s, fm in main_corr.items()}
    B = {s: {f: np.asarray(b, dtype=np.float64) for f, b in fm.items()} for s, fm in src_corr.items()}
    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in spliced.items()}
    ra = {s: frozen_runs(A[s]) for s in A}
    rb = {s: frozen_runs(B[s]) for s in B}
    n = 0
    n_cand = 0
    for s in out:
        if s not in A or s not in B:
            continue
        ks = sorted(set(out[s]) & set(A[s]) & set(B[s]))
        nfr = max(len(ks), 1)
        prev_a = prev_b = None
        for i, f in enumerate(ks):
            a, b = A[s][f], B[s][f]
            if modality(s) != 1:
                prev_a, prev_b = a, b
                continue
            if ra[s][f] >= K or rb[s][f] >= SPLICE_SRC_MAX_RUN:
                prev_a, prev_b = a, b
                continue
            iab = iou(a, b)
            if iab >= iab_max:
                prev_a, prev_b = a, b
                continue
            n_cand += 1
            x = feats(a, b, prev_a, prev_b, ra[s][f], rb[s][f], i / nfr, modality(s))
            p = float(predict_p(x[None, :], w, mu, sd)[0])
            if p < 0.5 - margin:
                out[s][f] = [float(v) for v in b]
                n += 1
            prev_a, prev_b = a, b
    return out, n, n_cand


def frozen_runs_tolerant(fm):
    """同 frozen_runs 但用 allclose 比較。

    ⚠️ 非贅餘：splice 用 exact（frozen_runs），而 selector 的特徵在 405 上是用
    allclose 算出來的（oof405_crossfit._frozen_runs(tolerant=True)）。特徵必須與
    fit 時同構，否則係數套在不同定義的 run 上。整數框下兩者等價，浮點框下不等價。
    """
    import numpy as np

    ks = sorted(fm)
    o, r = {}, 0
    for i, k in enumerate(ks):
        if i:
            r = r + 1 if bool(np.allclose(fm[k], fm[ks[i - 1]])) else 0
        else:
            r = 0
        o[k] = r
    return o


def apply_selector_v2(spliced, main_corr, src_corr, weights_path=SELECTOR_WEIGHTS, tau=None):
    """在 splice 之後、qhead 之前：迴歸預測 dIoU ≥ τ 的幀改採來源框（D085）。

    gate 與 v056 頭不同——**不限模態、不看 iab、不排除已 splice 的幀**，只要求
    非首幀且來源活著（src run < SPLICE_SRC_MAX_RUN），與 405 fit 時的 eligible
    定義一致。特徵取自校正後的主線／來源（非 splice 後的框），同 qhead 慣例。
    """
    import numpy as np
    from hsot.quality_head_v1 import feats, iou, modality  # noqa: F401
    from hsot.selector_v2 import predict_delta

    z = np.load(weights_path)
    w, mu, sd = z["w"], z["mu"], z["sd"]
    if tau is None:
        tau = float(z["tau"]) if "tau" in z.files else 0.05
    A = {s: {f: np.asarray(b, dtype=np.float64) for f, b in fm.items()} for s, fm in main_corr.items()}
    B = {s: {f: np.asarray(b, dtype=np.float64) for f, b in fm.items()} for s, fm in src_corr.items()}
    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in spliced.items()}
    ra = {s: frozen_runs_tolerant(A[s]) for s in A}
    rb = {s: frozen_runs_tolerant(B[s]) for s in B}
    n = 0
    for s in out:
        if s not in A or s not in B:
            continue
        ks = sorted(set(out[s]) & set(A[s]) & set(B[s]))
        nfr = max(len(ks), 1)
        prev_a = prev_b = None
        for i, f in enumerate(ks):
            a, b = A[s][f], B[s][f]
            if i == 0 or rb[s][f] >= SPLICE_SRC_MAX_RUN:
                prev_a, prev_b = a, b
                continue
            x = feats(a, b, prev_a, prev_b, ra[s][f], rb[s][f], i / nfr, modality(s))
            d = float(predict_delta(x[None, :], w, mu, sd)[0])
            if d >= tau:
                out[s][f] = [float(v) for v in b]
                n += 1
            prev_a, prev_b = a, b
    return out, n, float(tau)


def apply_third_leg_rescue(base, raw_main, raw_src, third_corr, K=SPLICE_K,
                           already_taken=None):
    """死區救援（D091）：main 凍結 ≥K 且來源也死 ⇒ 借第三腿（fresh full-frame SAM3）。

    這批幀 splice／selector／qhead 全部 gate 不到（三者都要求來源活著），所以是
    純加法、不與前面任何一層互斥。第四腿（fresh full SAMURAI）已於 v079 判死。

    ``already_taken``：前一腿已補過的 (seq, frame) 集合——多腿串接時傳入，避免後面的
    腿覆寫前面的（v079 當時是手動排除，這裡做成參數）。函式回傳本腿實際補的集合。
    """
    RM = {s: frozen_runs(raw_main[s]) for s in raw_main}
    RS = {s: frozen_runs(raw_src[s]) for s in raw_src}
    RC = {s: frozen_runs(third_corr[s]) for s in third_corr}
    taken = set() if already_taken is None else already_taken
    out = {s: {f: list(b) for f, b in fm.items()} for s, fm in base.items()}
    mine = set()
    n = 0
    for s in out:
        if s not in raw_main or s not in raw_src or s not in third_corr:
            continue
        for f in out[s]:
            if (s, f) in taken:
                continue
            if (f in raw_main[s] and f in raw_src[s] and f in third_corr[s]
                    and RM[s][f] >= K and RS[s][f] >= SPLICE_SRC_MAX_RUN
                    and RC[s][f] < SPLICE_SRC_MAX_RUN):
                out[s][f] = [float(v) for v in third_corr[s][f]]
                mine.add((s, f))
                n += 1
    return out, n, mine


def restore_first_frames(out, raw_main):
    """強制最終每序列首幀回復 raw main init，阻斷 splice/qhead 的旁路改寫。"""
    for s, fm in raw_main.items():
        if not fm:
            continue
        first = min(fm)
        out[s][first] = list(fm[first])
    return out


def validate_input_sets(main_order, source_order, sample_order):
    """production 三份 CSV 必須有完全相同的 unique ID set 與 row count。"""
    errors = []
    orders = {"main": main_order, "source": source_order, "sample": sample_order}
    for label, order in orders.items():
        n_unique = len(set(order))
        if n_unique != len(order):
            errors.append(f"{label}: 重複 ID {len(order) - n_unique} 列")
    reference = set(sample_order)
    for label in ("main", "source"):
        order = orders[label]
        ids = set(order)
        if len(order) != len(sample_order):
            errors.append(f"{label}/sample row count 不符：{len(order)} != {len(sample_order)}")
        if ids != reference:
            errors.append(
                f"{label}/sample exact-set 不符：缺 {len(reference - ids)} 列、"
                f"多 {len(ids - reference)} 列"
            )
    if set(main_order) != set(source_order):
        errors.append("main/source exact-set 不符")
    return _limited(errors)


def validate(out, order, sample_path=None, raw_main=None):
    errs = []
    keys = {(s, f) for s, fm in out.items() for f in fm}
    expected_keys = set()
    for rid in order:
        s, f = rid.rsplit("_", 1)
        expected_keys.add((s, int(f)))
    n_rows = sum(len(fm) for fm in out.values())
    if len(order) != len(set(order)):
        errs.append(f"輸出 order 含重複 ID {len(order) - len(set(order))} 列")
    if n_rows != len(order):
        errs.append(f"輸出 row count 不符：{n_rows} != {len(order)}")
    if keys != expected_keys:
        errs.append(f"輸出 exact-set 不符：缺 {len(expected_keys-keys)} 列、多 {len(keys-expected_keys)} 列")
    for s, fm in out.items():
        for f, (x, y, w, h) in fm.items():
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                errs.append(f"{s}_{f}: NaN/Inf")
            if x < 0 or y < 0:
                errs.append(f"{s}_{f}: 負 x/y ({x},{y})")
            if w <= 0 or h <= 0:
                errs.append(f"{s}_{f}: 非正 w/h ({w},{h})")
    if raw_main is not None:
        for s, fm in raw_main.items():
            if not fm:
                continue
            first = min(fm)
            got = out.get(s, {}).get(first)
            if got != fm[first]:
                errs.append(f"{s}_{first}: 首幀未等於 raw main init")
    if sample_path:
        try:
            _, sample_order = load(sample_path, "sample", validate_boxes=False)
            want = set(sample_order)
            if len(sample_order) != n_rows:
                errs.append(f"output/sample row count 不符：{n_rows} != {len(sample_order)}")
            if want != set(order):
                errs.append(
                    f"output/sample exact-set 不符：缺 {len(want-set(order))} 列、"
                    f"多 {len(set(order)-want)} 列"
                )
        except SubmissionValidationError as exc:
            errs.extend(exc.errors)
    return _limited(errs)


def selftest_sample_matches(sample_order, chain, seg):
    """(c)/(d) 專用：sample 的 ID 集合必須就是歷史 test75，否則 fail-fast 而不是 KeyError。

    9/7 的 drill 會把當天官方的新 sample 放進 repo，而 (c)/(d) 重現的是 test75 的凍結參照檔。
    兩者序列不同 ⇒ 舊碼在 write() 裡丟未捕捉的 KeyError，腳本當場死、機器 5 分鐘後自毀，
    症狀完全看不出是「sample 拿錯」（09-04 稽核 G11，已在本機 fakeroot 重現）。
    """
    want = {f"{s}_{f}" for s, fm in chain.items() for f in fm}
    got = set(sample_order)
    if want == got:
        return None
    only_sample = sorted(got - want)[:3]
    only_chain = sorted(want - got)[:3]
    return (
        f"🚨 自測 ({seg}) 無法執行：1_data/raw/sample_submisson.csv 的 ID 集合與凍結參照鏈不符"
        f"（sample {len(got)} 列 vs 參照 {len(want)} 列）。\n"
        f"   只在 sample 裡：{only_sample}\n"
        f"   只在參照裡：{only_chain}\n"
        f"   ⇒ 幾乎確定是 drill 把「本次要跑的 sample」錯當成「selftest 用的歷史 sample」覆蓋了。\n"
        f"   修法：drill 用 SELFTEST_SAMPLE（永遠拉 gDrive 的歷史 test75 sample）餵 selftest，\n"
        f"        SAMPLE 只傳給 run_ranking_b.py。"
    )


def write(out, order, path):
    """在目標目錄寫完並 fsync 後再 os.replace，中途失敗不會留半截成品。"""
    path = Path(path)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as fh:
            tmp = Path(fh.name)
            writer = csv.writer(fh)
            writer.writerow(SUBMISSION_COLUMNS)
            for i in order:
                s, f = i.rsplit("_", 1)
                writer.writerow([i] + [f"{v:g}" for v in out[s][int(f)]])
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _boxes_close(a, b, atol=1e-4):
    return all(abs(float(a[k]) - float(b[k])) <= atol for k in ("x", "y", "width", "height"))


def selftest():
    """(a) raw v023 + raw v012 → 位元級 v049；(b) 再套 v056 頭 → 浮點等於官方 v056。"""
    S = ROOT / "5_outputs/submissions"
    need = ["sub_v023_cropwiden55.csv", "sub_v012_e23b_sam21_ablation.csv", "sub_v049_src_v012_K6.csv"]
    for n in need:
        if not (S / n).is_file():
            print(f"🚨 自測缺檔：{n}", file=sys.stderr)
            return 2
    v056_path = S / "sub_v056_rn_iab03.csv"
    if not v056_path.is_file():
        print("🚨 自測缺檔：sub_v056_rn_iab03.csv", file=sys.stderr)
        return 2
    if not QHEAD_WEIGHTS.is_file():
        print(f"🚨 自測缺權重：{QHEAD_WEIGHTS}", file=sys.stderr)
        return 2

    main, order = load(S / need[0])
    src, _ = load(S / need[1])
    # 歷史 v049/v056 把校正套到首幀；僅 selftest 明示關閉新的 init 保護以重現舊檔。
    main_c = apply_correction(main, preserve_first=False)
    src_c = apply_correction(src, preserve_first=False)
    out, n = splice(main_c, src_c)
    tmp = Path("/tmp/_finalize_selftest_v049.csv")
    write(out, order, tmp)
    got = tmp.read_bytes()
    want = (S / need[2]).read_bytes()
    if got != want:
        g = list(csv.DictReader(open(tmp)))
        w_ = list(csv.DictReader(open(S / need[2])))
        diff = [(a["ID"], (a["x"], a["y"], a["width"], a["height"]), (b["x"], b["y"], b["width"], b["height"]))
                for a, b in zip(g, w_)
                if (a["x"], a["y"], a["width"], a["height"]) != (b["x"], b["y"], b["width"], b["height"])]
        print(f"🚨 自測失敗（v049）：{len(diff)}/{len(g)} 列不符（splice {n} 幀）", file=sys.stderr)
        for d in diff[:5]:
            print(f"   {d[0]}: got {d[1]} want {d[2]}", file=sys.stderr)
        return 2
    print(f"✅ 自測 (a) 通過：raw v023 + raw v012 → 位元級等於 v049（splice {n} 幀）")

    out56, nq, nc = apply_qhead_v056(out, main_c, src_c, K=SPLICE_K)
    g = {r["ID"]: r for r in csv.DictReader(open(v056_path))}
    diff = []
    for i in order:
        s, f = i.rsplit("_", 1)
        x, y, w, h = out56[s][int(f)]
        row = {"x": x, "y": y, "width": w, "height": h}
        if not _boxes_close(row, g[i]):
            diff.append((i, (x, y, w, h),
                         (g[i]["x"], g[i]["y"], g[i]["width"], g[i]["height"])))
    if nq != 110 or diff:
        print(f"🚨 自測失敗（v056）：替換 {nq}（要 110）、浮點不符 {len(diff)}、候選 {nc}",
              file=sys.stderr)
        for d in diff[:5]:
            print(f"   {d[0]}: got {d[1]} want {d[2]}", file=sys.stderr)
        return 2
    print(f"✅ 自測 (b) 通過：v049 + RedNIR 頭 → 浮點等於 v056（替換 {nq}／候選 {nc}）")

    # (c) 現行交付鏈：v078＝corr both → splice K6 → selector-v2 τ0.05 → qhead v056
    #     → 第三腿死區救援（LB 0.71666）。與 (a)(b) 的差別是首幀保護開著（production
    #     行為），故不能沿用上面的 preserve_first=False 中間結果，整條重跑。
    v078_path = S / "sub_v078_thirdleg_deadzone.csv"
    third_path = FROZEN_THIRD_LEG_FIXTURE
    sample_path = ROOT / "1_data/raw/sample_submisson.csv"
    missing = [str(p) for p in (v078_path, third_path, sample_path, SELECTOR_WEIGHTS)
               if not p.is_file()]
    if missing:
        print(f"⚠️ 自測 (c) 略過（缺 {len(missing)} 個輸入）：{missing[0]}")
        print("   ⚠️ 交付前必須讓 (c) 真的跑過——它是唯一覆蓋 selector／第三腿的自測")
        return _selftest_d()
    main_p = apply_correction(main)          # production：首幀保留
    src_p = apply_correction(src)
    chain, n2 = splice(main_p, src_p)
    chain, ns, tau_used = apply_selector_v2(chain, main_p, src_p)
    chain, nq2, _ = apply_qhead_v056(chain, main_p, src_p, K=SPLICE_K)
    raw_third, _ = load(third_path, "third-leg")
    chain, n3, _ = apply_third_leg_rescue(chain, main, src, apply_correction(raw_third))
    restore_first_frames(chain, main)
    tmp_c = Path(tempfile.gettempdir()) / "_finalize_selftest_v078.csv"
    _, sample_order = load(sample_path, "sample", validate_boxes=False)
    if (msg := selftest_sample_matches(sample_order, chain, "c")):
        print(msg, file=sys.stderr)
        return 2
    write(chain, sample_order, tmp_c)
    if tmp_c.read_bytes() != v078_path.read_bytes():
        g = list(csv.DictReader(open(tmp_c)))
        w_ = list(csv.DictReader(open(v078_path)))
        diff = [a["ID"] for a, b in zip(g, w_)
                if (a["x"], a["y"], a["width"], a["height"]) != (b["x"], b["y"], b["width"], b["height"])]
        print(f"🚨 自測失敗（v078 交付鏈）：{len(diff)}/{len(g)} 列不符 "
              f"[splice {n2}／selector {ns}@τ{tau_used:g}／qhead {nq2}／第三腿 {n3}]",
              file=sys.stderr)
        for i in diff[:5]:
            print(f"   {i}", file=sys.stderr)
        return 2
    if (n2, ns, nq2, n3) != (496, 7623, 110, 103):
        print(f"🚨 自測失敗（v078 分層計數）：得 {(n2, ns, nq2, n3)}，要 (496, 7623, 110, 103)",
              file=sys.stderr)
        return 2
    print(f"✅ 自測 (c) 通過：位元級等於 v078／LB 0.71666"
          f"（splice {n2}／selector {ns}@τ{tau_used:g}／qhead {nq2}／第三腿 {n3}）")
    return _selftest_d()


# (d) 的鎖定分層計數。與 (c) 的 (496, 7623, 110, 103) 不同是預期的：
# main 腿換成三窗 medoid ⇒ 每一層看到的候選都變了。
V090_LAYER_COUNTS = (513, 7652, 127, 37)


def _selftest_d():
    """(d) crop 窗三方共識鏈（D101）：medoid(A,B,C) → 交付後處理 → 位元級等於 v090／LB 0.71096。

    這是 ``rankB_deliver_v090`` profile 的位元級守護，對應 ``run_ranking_b.py`` 的
    ``medoid-crop-primary`` → ``finalize`` 兩步。三份輸入是 08-31 實跑的 crop-SAM3 merged
    輸出（窗 A/B/C），**都只用當次 run 自己的 full 輸出算窗**，故本段驗證的是後處理佈線，
    不是窗本身。⚠️ 輸入順序必須是 A,B,C——medoid 平手取第一個輸入。
    """
    S = ROOT / "5_outputs/submissions"
    D = ROOT / "5_outputs/cropwindow_ensemble_20260831"
    v090_path = S / "sub_v090_cropwin_medoid.csv"
    variants = [D / f"{tag}_main_merged.csv" for tag in ("A", "B", "C")]
    source_path = D / "A_source_merged.csv"      # 窗 A 的 SAMURAI crop-merged 腿
    third_path = D / "A_full_sam3.csv"           # 本次 run 的 full-frame SAM3＝第三腿
    sample_path = ROOT / "1_data/raw/sample_submisson.csv"
    missing = [str(x) for x in (*variants, source_path, third_path, v090_path,
                                sample_path, SELECTOR_WEIGHTS) if not x.is_file()]
    if missing:
        print(f"⚠️ 自測 (d) 略過（缺 {len(missing)} 個輸入）：{missing[0]}")
        print("   ⚠️ 若 9/7 走 rankB_deliver_v090（三窗共識），交付前必須讓 (d) 真的跑過")
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_ensemble_medoid as medoid_tool

    tmp_medoid = Path(tempfile.gettempdir()) / "_finalize_selftest_v090_medoid.csv"
    rc = medoid_tool.main([*[str(v) for v in variants], "--out", str(tmp_medoid)])
    if rc != 0:
        print("🚨 自測失敗（v090 medoid）：run_ensemble_medoid 非零結束", file=sys.stderr)
        return 2

    main, _ = load(tmp_medoid)
    src, _ = load(source_path)
    raw_third, _ = load(third_path, "third-leg")
    main_p, src_p = apply_correction(main), apply_correction(src)
    chain, n1 = splice(main_p, src_p)
    chain, ns, tau_used = apply_selector_v2(chain, main_p, src_p)
    chain, nq, _ = apply_qhead_v056(chain, main_p, src_p, K=SPLICE_K)
    chain, n3, _ = apply_third_leg_rescue(chain, main, src, apply_correction(raw_third))
    restore_first_frames(chain, main)
    tmp_d = Path(tempfile.gettempdir()) / "_finalize_selftest_v090.csv"
    _, sample_order = load(sample_path, "sample", validate_boxes=False)
    if (msg := selftest_sample_matches(sample_order, chain, "d")):
        print(msg, file=sys.stderr)
        return 2
    write(chain, sample_order, tmp_d)
    if tmp_d.read_bytes() != v090_path.read_bytes():
        g = list(csv.DictReader(open(tmp_d)))
        w_ = list(csv.DictReader(open(v090_path)))
        diff = [a["ID"] for a, b in zip(g, w_)
                if (a["x"], a["y"], a["width"], a["height"]) != (b["x"], b["y"], b["width"], b["height"])]
        print(f"🚨 自測失敗（v090 三窗共識鏈）：{len(diff)}/{len(g)} 列不符 "
              f"[splice {n1}／selector {ns}@τ{tau_used:g}／qhead {nq}／第三腿 {n3}]",
              file=sys.stderr)
        for i in diff[:5]:
            print(f"   {i}", file=sys.stderr)
        return 2
    if (n1, ns, nq, n3) != V090_LAYER_COUNTS:
        print(f"🚨 自測失敗（v090 分層計數）：得 {(n1, ns, nq, n3)}，要 {V090_LAYER_COUNTS}",
              file=sys.stderr)
        return 2
    print(f"✅ 自測 (d) 通過：位元級等於 v090／LB 0.71096"
          f"（splice {n1}／selector {ns}@τ{tau_used:g}／qhead {nq}／第三腿 {n3}）")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", help="主線 submission.csv（SAM3 + crop，未校正）")
    ap.add_argument("--source", help="來源 submission.csv（SAM2.1 + crop，未校正）")
    ap.add_argument("--out")
    ap.add_argument("--sample", help="sample_submisson.csv（production 必填；定義 canonical 輸出順序）")
    ap.add_argument("--K", type=int, default=SPLICE_K)
    ap.add_argument("--corr", choices=("both", "top-only", "none"), default="both",
                    help="座標校正範圍。both＝上+左（LB 最佳，+0.0142）；"
                         "top-only＝只套上緣（跨資料集信心較高，+0.0091）；none＝不套。"
                         "⚠️ Ranking B 選項，見 D072 風險分層")
    ap.add_argument("--no-correction", action="store_true", help="等同 --corr none（相容舊用法）")
    ap.add_argument("--qhead", choices=("v056", "none"), default="none",
                    help="splice 之後的 RedNIR 品質頭。production 預設 none；"
                         "v056 僅在明示選擇時啟用（Ranking A LB 0.71403）")
    ap.add_argument("--qhead-weights", default=str(QHEAD_WEIGHTS),
                    help="v056 logistic 權重 npz（w/mu/sd）")
    ap.add_argument("--selector", choices=("v2", "none"), default="none",
                    help="splice 之後、qhead 之前的 gain-weighted dIoU selector（D085）。"
                         "LB 實測 one-pass +0.00727／crop +0.00234")
    ap.add_argument("--selector-weights", default=str(SELECTOR_WEIGHTS),
                    help="selector-v2 固化權重 npz（w/mu/sd/tau），由 export_selector_weights.py 產生")
    ap.add_argument("--selector-tau", type=float, default=None,
                    help="覆寫權重檔內的 τ（僅供劑量掃描；交付用檔內值 0.05）")
    ap.add_argument("--third-leg",
                    help="死區救援來源 CSV（D091）：fresh full-frame SAM3 腿的 submission.csv。"
                         "只補 main 凍結≥K 且來源亦死的幀，LB 實測 +0.00029")
    ap.add_argument("--fourth-leg",
                    help="第二個死區救援來源（D095）：例如 tracker 微調版的 full-frame SAM3。"
                         "只補【第三腿補不到】的死區幀（第三腿優先，本腿撿剩）。"
                         "⚠️ 微調權重在健康序列上實測 −0.054，**不可當主線**；"
                         "死區是它唯一必然安全的落點（那些幀主線與來源都已死）")
    ap.add_argument("--selftest", action="store_true", help="對已知答案自測（交付前必跑）")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if not (a.main and a.source and a.sample and a.out):
        ap.error("--main／--source／--sample／--out 皆為 production 必填（或改用 --selftest）")
    if a.K < 1:
        ap.error("--K 必須 >= 1")

    try:
        raw_main, main_order = load(a.main, "main")
        raw_src, source_order = load(a.source, "source")
        _, sample_order = load(a.sample, "sample", validate_boxes=False)
    except SubmissionValidationError as exc:
        print("❌ 輸入驗證未過：", file=sys.stderr)
        for e in exc.errors:
            print(f"    - {e}", file=sys.stderr)
        sys.exit(2)
    input_errs = validate_input_sets(main_order, source_order, sample_order)
    if input_errs:
        print("❌ 輸入驗證未過：", file=sys.stderr)
        for e in input_errs:
            print(f"    - {e}", file=sys.stderr)
        sys.exit(2)

    mode = "none" if a.no_correction else a.corr
    top = CORR_TOP_PX if mode in ("both", "top-only") else 0.0
    left = CORR_LEFT_PX if mode == "both" else 0.0
    if top or left:
        main_d = apply_correction(raw_main, top, left)
        src_d = apply_correction(raw_src, top, left)
        print(f"[1] 座標校正 --corr {mode}：上 +{top:g}px、左 +{left:g}px（clamp≥0）已套用於兩套輸出；首幀 init 保留")
    else:
        main_d = {s: {f: list(b) for f, b in fm.items()} for s, fm in raw_main.items()}
        src_d = {s: {f: list(b) for f, b in fm.items()} for s, fm in raw_src.items()}
        print("[1] ⚠️ 未套座標校正（--corr none）——放棄 +0.0142，僅在確信新資料無偏移時使用")
    spliced, n = splice(main_d, src_d, a.K)
    tot = sum(len(v) for v in main_d.values())
    print(f"[2] 跨底座凍結救援：K={a.K}，替換 {n} 幀（{n/tot:.2%} of {tot}）")
    step = 3
    if a.selector == "v2":
        spliced, ns, tau_used = apply_selector_v2(
            spliced, main_d, src_d, Path(a.selector_weights), a.selector_tau)
        print(f"[{step}] selector-v2（D085）：τ={tau_used:g}，再換 {ns} 幀（{ns/tot:.2%}）")
        step += 1
    if a.qhead == "v056":
        out, nq, nc = apply_qhead_v056(spliced, main_d, src_d, Path(a.qhead_weights), K=a.K)
        print(f"[{step}] RedNIR 品質頭 v056：iab<{QHEAD_IAB_MAX:g}，候選 {nc}，換 B {nq} 幀")
        step += 1
    else:
        out = spliced
        print(f"[{step}] 未套品質頭（--qhead none）")
        step += 1
    if a.third_leg:
        try:
            raw_third, _ = load(a.third_leg, "third-leg")
        except SubmissionValidationError as exc:
            print("❌ 第三腿輸入驗證未過：", file=sys.stderr)
            for e in exc.errors:
                print(f"    - {e}", file=sys.stderr)
            sys.exit(2)
        third_d = apply_correction(raw_third, top, left) if (top or left) else \
            {s: {f: list(b) for f, b in fm.items()} for s, fm in raw_third.items()}
        out, n3, taken3 = apply_third_leg_rescue(out, raw_main, raw_src, third_d, K=a.K)
        print(f"[{step}] 第三腿死區救援（D091）：補 {n3} 幀（main 凍結≥{a.K} 且來源亦死）")
        step += 1
    if a.fourth_leg:
        if not a.third_leg:
            ap.error("--fourth-leg 需與 --third-leg 併用（本腿只撿第三腿補不到的死區幀）")
        try:
            raw_fourth, _ = load(a.fourth_leg, "fourth-leg")
        except SubmissionValidationError as exc:
            print("❌ 第四腿輸入驗證未過：", file=sys.stderr)
            for e in exc.errors:
                print(f"    - {e}", file=sys.stderr)
            sys.exit(2)
        fourth_d = apply_correction(raw_fourth, top, left) if (top or left) else \
            {s: {f: list(b) for f, b in fm.items()} for s, fm in raw_fourth.items()}
        out, n4, _ = apply_third_leg_rescue(out, raw_main, raw_src, fourth_d, K=a.K,
                                            already_taken=taken3)
        print(f"[{step}] 第四腿死區救援（D095）：再補 {n4} 幀（第三腿補不到的）")
        step += 1
    step_v = step

    restore_first_frames(out, raw_main)
    errs = validate(out, sample_order, a.sample, raw_main=raw_main)
    if errs:
        print(f"[{step_v}] ❌ 驗證未過：", file=sys.stderr)
        for e in errs:
            print(f"    - {e}", file=sys.stderr)
        sys.exit(2)
    write(out, sample_order, a.out)
    print(f"[{step_v}] ✅ 驗證通過（{tot} 列）→ {a.out}")


if __name__ == "__main__":
    main()
