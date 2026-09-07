import numpy as np
from scripts.calibrate_fusion import grid_search_fusion_params


class TestFusionCalibration:
    """Ensure fusion grid search selects optimal alpha and threshold."""

    def test_alpha_grid_search_optimizes_eer(self):
        # Synthetic scenario: spatial has EER 15%, temporal has EER 10%,
        # fused has complementary errors with EER < 5%
        rng = np.random.default_rng(42)
        n = 200
        # Spatial scores: genuine ~ N(0.8, 0.1), impostor ~ N(0.5, 0.1)
        s_gen = rng.normal(0.8, 0.1, n)
        s_imp = rng.normal(0.5, 0.1, n)

        # Temporal scores: genuine ~ N(0.85, 0.08), impostor ~ N(0.45, 0.08)
        t_gen = rng.normal(0.85, 0.08, n)
        t_imp = rng.normal(0.45, 0.08, n)

        best_alpha, best_tau, best_eer = grid_search_fusion_params(
            s_gen, s_imp, t_gen, t_imp, alpha_steps=11
        )
        assert 0.0 <= best_alpha <= 1.0
        assert 0.0 <= best_tau <= 1.0
        assert best_eer < 0.05
