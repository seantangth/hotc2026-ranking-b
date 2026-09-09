"""run_ensemble_medoid 的行為鎖定（D096）。

重點鎖三件事：
1. 跑飛的那一份會被淘汰（N=3 時退化為多數決）——這是本工具存在的理由；
2. **N=2 時 medoid 恆等於現任**——08-30 的 v084 有 40 支序列踩到這個坑（只有兩份
   獨立答案），該批等於沒被驗證。這條測試把它變成明文契約，避免下次再誤設計；
3. 首幀（官方 init box）不得被改動。
"""
import csv

import pytest

from run_ensemble_medoid import iou, main, medoid

HEADER = ["ID", "x", "y", "width", "height"]


def write(path, rows):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for key, box in rows.items():
            w.writerow([key, *box])
    return path


def read(path):
    with path.open(newline="") as fh:
        r = csv.reader(fh)
        next(r)
        return {row[0]: tuple(int(v) for v in row[1:]) for row in r}


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_medoid_drops_the_diverged_run():
    """兩份互相貼合、一份跑到別的物體 ⇒ 跑飛的那份出局。"""
    good_a, good_b, flown = (10, 10, 20, 20), (11, 10, 20, 20), (300, 300, 20, 20)
    box, idx = medoid([flown, good_a, good_b])
    assert box in (good_a, good_b) and idx != 0


def test_medoid_ties_go_to_incumbent():
    a, b = (0, 0, 10, 10), (50, 50, 10, 10)
    assert medoid([a, b])[1] == 0          # N=2：兩者分數相同 → 取現任
    assert medoid([b, a])[1] == 0


def test_two_runs_are_always_a_noop(tmp_path):
    """N=2 的契約：輸出必然等於現任，一列都不會變。"""
    inc = {"seq_1": (1, 1, 5, 5), "seq_2": (2, 2, 5, 5)}
    other = {"seq_1": (1, 1, 5, 5), "seq_2": (99, 99, 5, 5)}
    p1 = write(tmp_path / "a.csv", inc)
    p2 = write(tmp_path / "b.csv", other)
    out = tmp_path / "out.csv"
    assert main([str(p1), str(p2), "--out", str(out)]) == 0
    assert read(out) == inc


def test_three_runs_majority_wins(tmp_path):
    inc = {"seq_1": (1, 1, 5, 5), "seq_2": (99, 99, 5, 5)}
    run_d = {"seq_1": (1, 1, 5, 5), "seq_2": (2, 2, 5, 5)}
    run_e = {"seq_1": (1, 1, 5, 5), "seq_2": (2, 2, 5, 5)}
    out = tmp_path / "out.csv"
    assert main([str(write(tmp_path / "a.csv", inc)),
                 str(write(tmp_path / "d.csv", run_d)),
                 str(write(tmp_path / "e.csv", run_e)),
                 "--out", str(out)]) == 0
    got = read(out)
    assert got["seq_1"] == (1, 1, 5, 5)      # 三者全同
    assert got["seq_2"] == (2, 2, 5, 5)      # 現任被多數決推翻


def test_first_frame_must_not_move(tmp_path):
    """首幀＝官方 init box，任何共識規則都不得改動它。"""
    inc = {"seq_1": (1, 1, 5, 5), "seq_2": (1, 1, 5, 5)}
    other = {"seq_1": (80, 80, 5, 5), "seq_2": (1, 1, 5, 5)}
    out = tmp_path / "out.csv"
    with pytest.raises(SystemExit, match="首幀"):
        main([str(write(tmp_path / "a.csv", inc)),
              str(write(tmp_path / "d.csv", other)),
              str(write(tmp_path / "e.csv", other)),
              "--out", str(out)])


def test_id_set_mismatch_is_fatal(tmp_path):
    out = tmp_path / "out.csv"
    with pytest.raises(SystemExit, match="ID 集合不一致"):
        main([str(write(tmp_path / "a.csv", {"seq_1": (1, 1, 5, 5)})),
              str(write(tmp_path / "d.csv", {"seq_2": (1, 1, 5, 5)})),
              "--out", str(out)])


def test_row_order_follows_incumbent(tmp_path):
    inc = {"seq_2": (1, 1, 5, 5), "seq_1": (1, 1, 5, 5)}
    out = tmp_path / "out.csv"
    main([str(write(tmp_path / "a.csv", inc)),
          str(write(tmp_path / "d.csv", inc)),
          "--out", str(out)])
    assert list(read(out)) == ["seq_2", "seq_1"]
