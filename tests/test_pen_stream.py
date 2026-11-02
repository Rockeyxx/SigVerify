"""Raw pen strokes to the model input: clock repair, the 10 ms grid, y up, still-pen holds."""
import numpy as np
import pytest

from config.settings import TEMPORAL_INPUT_FEATURES
from src.data.dataset import trajectory_gate
from src.data.parsers import MAX_GAP_MS
from src.data.pen_stream import (device_period, repair_clock, resample_strokes,
                                 strokes_to_trajectory)

STEP = 4.0   # the XPPen's 250 Hz report interval in ms (ENG_REPORT G4)
needs_dt = pytest.mark.skipif(TEMPORAL_INPUT_FEATURES < 4,
                              reason="no dt channel under the current contract")


def _stroke(t0, n, x0=0.0, y0=0.0, vx=0.5, vy=0.2, p=8000.0):
    t = t0 + STEP * np.arange(n)
    return np.column_stack([t, x0 + vx * (t - t0), y0 + vy * (t - t0), np.full(n, p)])


class TestRepairClock:
    def test_unwraps_a_32_bit_millisecond_counter(self):
        t = np.array([2.0 ** 32 - 8, 2.0 ** 32 - 4, 0.0, 4.0])
        assert np.allclose(np.diff(repair_clock(t, STEP)), STEP)

    def test_spreads_samples_that_share_a_timestamp(self):
        assert np.allclose(repair_clock(np.array([0.0, 0.0, 0.0, 12.0]), STEP), [0, 4, 8, 12])

    def test_a_backward_step_becomes_one_period(self):
        assert np.allclose(repair_clock(np.array([0.0, 4.0, 2.0, 6.0]), STEP), [0, 4, 8, 12])

    def test_long_forward_gaps_are_kept(self):
        assert np.allclose(repair_clock(np.array([0.0, 4.0, 1504.0]), STEP), [0, 4, 1504])


class TestResampleStrokes:
    def test_250_hz_lands_on_an_exact_10_ms_grid(self):
        t, *_ = resample_strokes([_stroke(0.0, 51)], pressure_max=16383)
        assert np.allclose(np.diff(t), 10.0)
        assert t[0] == 0.0 and t[-1] == 200.0

    def test_y_is_flipped_to_grow_upward(self):
        _, _, y, _ = resample_strokes([_stroke(0.0, 51, vy=1.0)], pressure_max=16383)
        assert y[-1] < y[0]   # moving down the screen is moving down in pen coordinates

    def test_pressure_is_scaled_by_the_device_maximum(self):
        *_, p = resample_strokes([_stroke(0.0, 51, p=8191.5)], pressure_max=16383)
        assert np.allclose(p, 0.5)

    def test_a_still_pen_is_held_not_drifted(self):
        """The XPPen sends nothing while the pen is still (gaps up to 626 ms, G4)."""
        still = np.array([[0.0, 0, 0, 5000], [4.0, 0, 0, 5000], [8.0, 0, 0, 5000],
                          [308.0, 10, 0, 5000], [312.0, 10.5, 0, 5000]])
        t, x, _, _ = resample_strokes([still], pressure_max=16383)
        assert np.all(x[(t > 8) & (t < 308)] == 0.0)

    def test_a_lift_drops_pen_up_grid_points(self):
        t, *_ = resample_strokes([_stroke(0.0, 26), _stroke(150.0, 26)], pressure_max=16383)
        assert not np.any((t > 100) & (t < 150))

    def test_a_dot_shorter_than_one_step_keeps_one_point(self):
        dot = np.array([[13.0, 5, 5, 4000], [15.0, 5, 5, 4000]])
        t, *_ = resample_strokes([_stroke(0.0, 3), dot], pressure_max=16383)
        assert 13.0 in t

    def test_no_samples_is_an_error(self):
        with pytest.raises(ValueError):
            resample_strokes([], pressure_max=16383)

    def test_device_period_is_the_median_report_interval(self):
        assert device_period([_stroke(0.0, 20)]) == STEP


class TestStrokesToTrajectory:
    @needs_dt
    def test_a_lift_shows_as_its_duration_in_dt(self):
        traj = strokes_to_trajectory([_stroke(0.0, 26), _stroke(150.0, 26)], pressure_max=16383)
        assert np.isclose(traj[11, 3], 50.0 / MAX_GAP_MS)   # stroke 2 starts 50 ms after stroke 1

    @needs_dt
    def test_output_passes_the_trajectory_gate(self):
        strokes = [_stroke(0.0, 60), _stroke(300.0, 60, x0=40, vy=-0.3), _stroke(600.0, 40, x0=80)]
        traj = strokes_to_trajectory(strokes, pressure_max=16383)
        passed, details = trajectory_gate(traj)
        assert passed, details
        assert traj.shape[1] == TEMPORAL_INPUT_FEATURES
        assert np.isclose(np.median(traj[:, 3]), 10.0 / MAX_GAP_MS)
