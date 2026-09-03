import numpy as np
import pytest
from src.data.renderer import render_trajectory_to_image


class TestTrajectoryRenderer:
    """Ensure trajectories rasterize into valid spatial input tensors."""

    def _sample_trajectory(self, num_points=100):
        # Parametric spiral signature trajectory [L, 4]
        t = np.linspace(0, 4 * np.pi, num_points)
        x = (t * np.cos(t)).astype(np.float32)
        y = (t * np.sin(t)).astype(np.float32)
        p = (0.3 + 0.6 * (np.sin(t) ** 2)).astype(np.float32)
        dt = np.full(num_points, 0.01, dtype=np.float32)
        # standardize x, y
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)
        return np.stack([x, y, p, dt], axis=1)

    def test_output_tensor_shape_and_dtype(self):
        traj = self._sample_trajectory(100)
        img = render_trajectory_to_image(traj, target_size=(220, 155))
        assert isinstance(img, np.ndarray)
        assert img.shape == (1, 155, 220)
        assert img.dtype == np.float32

    def test_inverted_polarity_background_is_zero_ink_is_positive(self):
        traj = self._sample_trajectory(100)
        img = render_trajectory_to_image(traj, target_size=(220, 155))
        # Background must be 0.0
        assert img[0, 0, 0] == 0.0
        # Canvas corners should be empty background
        assert img[0, 0, -1] == 0.0
        assert img[0, -1, 0] == 0.0
        # Ink strokes must be present and bounded in [0, 1]
        assert img.max() > 0.5
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    def test_pen_up_segments_do_not_draw_connecting_lines(self):
        # Two separated dots with P=0 in between
        traj = np.array([
            [-1.0, 0.0, 0.8, 0.01],
            [-0.9, 0.0, 0.8, 0.01],
            [0.0, 0.0, 0.0, 0.05],  # Pen lift
            [0.9, 0.0, 0.8, 0.01],
            [1.0, 0.0, 0.8, 0.01],
        ], dtype=np.float32)
        img = render_trajectory_to_image(traj, target_size=(220, 155))
        # Midpoint horizontally (x=110) should have no ink because pen was lifted
        mid_x = 110
        assert (img[0, :, mid_x] == 0.0).all()

    def test_passes_spatial_quality_gate(self):
        from src.data.dataset import quality_gate
        traj = self._sample_trajectory(200)
        img = render_trajectory_to_image(traj, target_size=(220, 155))
        passed, details = quality_gate(img)
        assert passed, f"Rendered image failed quality gate: {details}"


class TestPenLiftFromTheClock:
    """
    A lift shows up as a gap in the clock, not only as vanishing pressure.

    SigComp2011 records pen-up travel as force = 0, but SVC2004 and DeepSignDB
    drop those samples and leave only the time gap. Judging that gap against an
    absolute number of seconds is wrong in both directions -- at the 10 Hz used
    by some fixtures a 0.05 s cutoff marks *every* sample as a lift and renders a
    blank canvas, while at SigComp2011's 200 Hz it catches none. The threshold
    therefore scales to each signature's own median sampling period.
    """

    def _two_strokes(self, dt_gap, dt_normal=0.01, pressure=0.8):
        """Two horizontal strokes; the clock jumps by `dt_gap` between them."""
        xs = np.concatenate([np.linspace(-1.0, -0.2, 12), np.linspace(0.2, 1.0, 12)])
        traj = np.zeros((len(xs), 4), dtype=np.float32)
        traj[:, 0] = xs
        traj[:, 1] = 0.0
        traj[:, 2] = pressure          # pen never leaves the surface by pressure
        traj[:, 3] = dt_normal
        traj[12, 3] = dt_gap           # the lift, expressed only in the clock
        return traj

    def test_a_clock_gap_breaks_the_stroke(self):
        img = render_trajectory_to_image(self._two_strokes(dt_gap=0.5))
        # Mid-canvas column sits between the two strokes; nothing should bridge it.
        assert img[0, :, 110].max() == 0.0
        assert img.max() > 0.5, "both strokes should still be drawn"

    def test_uniform_sampling_draws_one_continuous_stroke(self):
        """No gap means no lift, however slow the capture."""
        img = render_trajectory_to_image(self._two_strokes(dt_gap=0.01))
        assert img[0, :, 110].max() > 0.0

    def test_slow_capture_is_not_read_as_all_pen_up(self):
        """
        10 Hz uniform sampling: every dt is 0.1 s, far above any fixed cutoff,
        yet none of it is a lift. This is the case that rendered a blank canvas.
        """
        traj = self._two_strokes(dt_gap=0.1, dt_normal=0.1)
        img = render_trajectory_to_image(traj)
        assert img.max() > 0.5, "a slow but uniform capture must still render ink"
