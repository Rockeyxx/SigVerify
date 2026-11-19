"""tools/research/campaign_eval.py: product order and the step-up rule, on made-up embeddings."""
import numpy as np

from config.settings import STEP_UP_ACCEPT, STEP_UP_REJECT, STEP_UP_TWO
from tools.research.campaign_eval import product_rows, step_up_accepts


def test_product_order_uses_the_first_five_as_references_and_scores_the_rest():
    genuine = np.arange(12, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    forged = np.full((2, 3), 100.0, np.float32)
    rows = product_rows({"w": {"genuine": (genuine, genuine), "forged": (forged, forged)}})
    scan, pen = rows["w"]["genuine"]
    assert len(scan) == 7 and np.array_equal(scan, pen)
    # probe 6 (value 5) is nearest reference 5 (value 4): distance sqrt(3)
    assert np.isclose(scan[0], 1 / (1 + np.sqrt(3)))
    assert scan[0] > scan[-1]                     # later probes are farther from the references
    assert len(rows["w"]["forged"][0]) == 2


def test_step_up_rule_matches_verify():
    assert step_up_accepts(STEP_UP_ACCEPT, 0.0)
    assert not step_up_accepts(STEP_UP_REJECT - 1e-6, 1.0)
    mid = (STEP_UP_REJECT + STEP_UP_ACCEPT) / 2
    assert step_up_accepts(mid, 2 * STEP_UP_TWO - mid)
    assert not step_up_accepts(mid, 2 * STEP_UP_TWO - mid - 1e-3)


def test_faint_greys_the_paper_and_softens_the_ink():
    from tools.research.campaign_eval import faint
    img = np.full((40, 40), 255, np.uint8)
    img[15:25, 5:35] = 0
    out = faint(img)
    assert out[0, 0] == 183                                  # paper x 183/255
    assert 50 <= out[20, 20] <= 60                           # ink: 183 - 183 * 0.7 = 54.9
    assert out.dtype == np.uint8


def test_held_out_sets_tau_on_train_writers_only():
    from tools.research.campaign_eval import held_out
    good = {"genuine": (np.full(5, 0.9), np.full(5, 0.9)), "forged": (np.full(2, 0.5), np.full(2, 0.5))}
    odd = {"genuine": (np.full(5, 0.3), np.full(5, 0.3)), "forged": (np.full(2, 0.1), np.full(2, 0.1))}
    rows = {"a": good, "b": good, "c": odd}
    h = held_out(rows, {"train": ["a", "b"], "test": ["c"]}, 0.4, 0.8)
    assert h["train"] == 2 and h["test"] == 1
    assert 0.5 <= h["tau"] <= 0.9 + 1e-9                     # from a and b, not from c
    assert h["frr"] == 100.0 and h["far_shipped"] == 0.0
