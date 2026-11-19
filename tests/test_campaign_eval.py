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
