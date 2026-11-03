"""
One spatial dataset holding scans and pen trajectories at the same time.

`SpatialSignatureDataset` used to accept only paths, so an online corpus could
never contribute to the spatial branch even though `render_trajectory_to_image`
has produced the branch's exact input contract since the fusion work. This suite
gates the mixed loader.

**The whole risk is in one line of the fork.** A trajectory `[L, 4]` and a raw
scan `[H, W]` are both 2-D float arrays, so nothing about "is this an image or a
motion record" is visible in the type. Guessing wrong is silent in both
directions: a scan handed to the renderer draws a scribble out of pixel
intensities, and a trajectory handed to `preprocess_image` gets resized onto
220x155 and inverted, producing a perfectly well-formed tensor that means
nothing. Four earlier defects had exactly that shape:
plausible output, no exception, green suite. `TestTheDisambiguationRule` below is
the part of this file that has to survive; the rest is geometry.
"""
import numpy as np
import pytest
import torch

import cv2

import src.data.dataset as ds
from config.settings import (
    MIN_BBOX_PIXELS,
    MIN_TRAJECTORY_POINTS,
    SPATIAL_INPUT_SIZE,
    TEMPORAL_CHANNEL_WIDTHS,
    TEMPORAL_INPUT_FEATURES,
)
from src.data.dataset import SpatialSignatureDataset, preprocess_image
from src.data.renderer import render_trajectory_to_image

WIDTH, HEIGHT = SPATIAL_INPUT_SIZE


def _trajectory(num_points=120, columns=TEMPORAL_INPUT_FEATURES, phase=0.0):
    """
    A normalized `[L, C]` pen trajectory, shaped like what the parsers emit.

    Width follows the capture contract rather than a literal, so this fixture
    moves with `TEMPORAL_CHANNELS` instead of agreeing with a stale gate — the
    failure mode seen before, where every fixture in the suite built
    3-column trajectories and therefore agreed with a gate the shipped weights
    had already outgrown.
    """
    t = np.linspace(0, 4 * np.pi, num_points) + phase
    x = (t * np.cos(t)).astype(np.float32)
    y = (t * np.sin(t)).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    channels = [x, y, (0.3 + 0.6 * np.sin(t) ** 2).astype(np.float32)]
    while len(channels) < columns:
        channels.append(np.full(num_points, 0.1, dtype=np.float32))
    return np.stack(channels[:columns], axis=1).astype(np.float32)


def _scan_array(h=300, w=400):
    """A raw scan exactly as `cv2.imread(..., IMREAD_GRAYSCALE)` returns it."""
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.line(img, (40, 150), (360, 160), 0, 4)
    cv2.line(img, (90, 90), (220, 240), 0, 3)
    cv2.ellipse(img, (250, 150), (70, 40), 20, 0, 300, 0, 3)
    return img


def _scan_path(tmp_path, name="scan.png", **kwargs):
    path = str(tmp_path / name)
    cv2.imwrite(path, _scan_array(**kwargs))
    return path


def _item(pairs, labels=None, **kwargs):
    labels = [1.0] * len(pairs) if labels is None else labels
    return SpatialSignatureDataset(pairs, labels, **kwargs)[0]


class TestEveryPairKindReachesTheModel:
    """Three sample kinds, freely mixed inside one pair."""

    def test_path_pair(self, tmp_path):
        a, b = _scan_path(tmp_path, "a.png"), _scan_path(tmp_path, "b.png")
        first, second, label = _item([(a, b)])
        assert first.shape == (1, HEIGHT, WIDTH)
        assert second.shape == (1, HEIGHT, WIDTH)
        assert label.item() == 1.0

    def test_trajectory_pair(self):
        first, second, label = _item([(_trajectory(), _trajectory(phase=0.4))])
        assert first.shape == (1, HEIGHT, WIDTH)
        assert second.shape == (1, HEIGHT, WIDTH)
        assert first.max() > 0.5 and second.max() > 0.5, "neither side drew ink"
        assert label.item() == 1.0

    def test_cross_modal_pair(self, tmp_path):
        """
        The point of the whole change: a scanned reference against a tablet
        capture, inside one pair, with one label.
        """
        first, second, label = _item([(_scan_path(tmp_path), _trajectory())], [0.0])
        assert first.shape == second.shape == (1, HEIGHT, WIDTH)
        assert first.max() > 0.0 and second.max() > 0.0
        assert label.item() == 0.0

    def test_raw_scan_array_is_treated_exactly_like_its_path(self, tmp_path):
        """An array and the file it came from must reach the model identically."""
        path = _scan_path(tmp_path)
        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        from_path, from_array, _ = _item([(path, array)])
        torch.testing.assert_close(from_path, from_array)

    def test_canonical_array_passes_through_untouched(self):
        """
        The renderer's own output is already the contract. Sending it through
        `preprocess_image` would invert it a second time and train the branch on
        photographic negatives.
        """
        rendered = render_trajectory_to_image(_trajectory())
        first, _, _ = _item([(rendered, rendered)])
        np.testing.assert_array_equal(first.numpy(), rendered)

    def test_the_three_tuple_contract_is_unchanged(self, tmp_path):
        item = _item([(_scan_path(tmp_path), _trajectory())])
        assert len(item) == 3
        assert all(isinstance(part, torch.Tensor) for part in item)


class TestRenderedSideMatchesTheScanContract:
    """A rendered trajectory must be indistinguishable in form from a scan."""

    def test_geometry_comes_from_config(self):
        """
        Read from `SPATIAL_INPUT_SIZE`, never pinned to 155x220. The geometry is
        a measured choice that already moved once (256x256 square-pad lost on
        every corpus), so a hardcoded assertion would fail the next measurement
        rather than the next bug.
        """
        first, second, _ = _item([(_trajectory(), _scan_array())])
        assert first.shape == (1, HEIGHT, WIDTH) == second.shape

    def test_polarity_is_inverted_background_zero(self):
        """
        Background 0, ink high — worth 4.86% to 0.00% EER on CEDAR. A rendered
        side that arrived un-inverted would put half the batch in the opposite
        polarity from the other half.
        """
        first, _, _ = _item([(_trajectory(), _trajectory())])
        image = first.numpy()
        assert image[0, 0, 0] == 0.0
        assert image[0, 0, -1] == 0.0
        assert image[0, -1, 0] == 0.0
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert image.max() > 0.5, "ink must be high, not merely present"

    def test_dtype_is_float32(self):
        first, second, label = _item([(_trajectory(), _scan_array())])
        assert first.dtype == torch.float32
        assert second.dtype == torch.float32
        assert label.dtype == torch.float32


class TestTheDisambiguationRule:
    """
    A trajectory and a scan are both 2-D float arrays. The fork keys on the
    **column count**, which is the one property whose two admissible ranges
    cannot overlap:

    * a trajectory carries `TEMPORAL_CHANNEL_WIDTHS` columns, 3 to 7. Widths 6
      and 7 are the research-only tilt modes; the capture contract stays 4
      under AGENTS.md Drift Prevention;
    * a scan is at least `MIN_BBOX_PIXELS` (64) wide, because the ink bounding
      box cannot exceed the canvas and anything under that floor already fails
      `quality_gate`.

    Everything in between is **refused**, not guessed at. Value range and dtype
    were rejected as the discriminator: a `[0, 1]` float scan is a legitimate
    canonical array, a trajectory's pressure and `dt` columns are also in
    `[0, 1]`, and a trajectory still in raw device units looks exactly like a
    scan by magnitude.

    Delete the fork and these tests fail. That is the point of them.
    """

    def test_a_trajectory_is_rendered_not_resized(self):
        """
        The failure this file exists for. `preprocess_image` accepts an `[L, 4]`
        array without complaint and returns a well-formed tensor built from
        coordinate values read as pixel intensities.
        """
        trajectory = _trajectory()
        rendered, _, _ = _item([(trajectory, trajectory)])

        np.testing.assert_array_equal(rendered.numpy(),
                                      render_trajectory_to_image(trajectory))
        resized = preprocess_image(trajectory)
        assert not np.allclose(rendered.numpy(), resized), (
            "the trajectory went through the scan path — this is the silent "
            "defect, not a cosmetic one")

    def test_a_scan_is_resized_not_rendered(self):
        """The mirror image: a scan must never reach the rasterizer."""
        scan = _scan_array()
        first, _, _ = _item([(scan, scan)])
        np.testing.assert_array_equal(first.numpy(), preprocess_image(scan))

    def test_the_column_band_comes_from_the_capture_contract(self):
        """
        Not a literal 4. `TEMPORAL_CHANNELS` is the single knob for the capture
        width, and a hardcoded count in a gate has already made the shipped
        model unreachable once.
        """
        assert ds._TRAJECTORY_MIN_COLUMNS == min(TEMPORAL_CHANNEL_WIDTHS.values())
        assert ds._TRAJECTORY_MAX_COLUMNS == max(TEMPORAL_CHANNEL_WIDTHS.values())

    @pytest.mark.parametrize("columns", sorted(set(TEMPORAL_CHANNEL_WIDTHS.values())))
    def test_every_contract_width_routes_to_the_renderer(self, columns):
        """
        3-, 4- and 5-column trajectories all render, whichever width happens to
        be shipping. Keying on the *current* width alone would send the other
        two down the scan path the day the contract moved.
        """
        trajectory = _trajectory(columns=columns)
        first, _, _ = _item([(trajectory, trajectory)])
        np.testing.assert_array_equal(first.numpy(),
                                      render_trajectory_to_image(trajectory))

    def test_a_column_count_between_the_two_kinds_is_refused(self):
        """
        12 columns is too wide for any capture contract and far too narrow for a
        scan. There is no correct guess, so it raises.
        """
        with pytest.raises(ValueError, match="Cannot interpret"):
            _item([(np.zeros((200, 12), dtype=np.float32),) * 2])

    def test_a_transposed_trajectory_is_refused(self):
        """
        `[C, L]` instead of `[L, C]`. Its column count reads as a scan, and
        resizing four rows onto a 155-row canvas returns a plausible smear. The
        row floor is what catches it.
        """
        with pytest.raises(ValueError, match="Cannot interpret"):
            _item([(_trajectory().T,) * 2])

    def test_a_scan_below_the_resolution_floor_is_refused(self):
        small = np.full((MIN_BBOX_PIXELS - 1, MIN_BBOX_PIXELS - 1), 255, np.uint8)
        with pytest.raises(ValueError, match="Cannot interpret"):
            _item([(small, small)])

    def test_a_trajectory_too_short_to_draw_is_refused(self):
        """
        Under `MIN_TRAJECTORY_POINTS` the renderer returns a near-blank canvas
        rather than raising, and a blank "signature" trains as a valid sample.
        """
        short = _trajectory(num_points=MIN_TRAJECTORY_POINTS - 1)
        with pytest.raises(ValueError, match="points"):
            _item([(short, short)])

    def test_a_two_column_array_is_refused(self):
        """Narrower than any contract: X and Y with no pressure is not a sample."""
        with pytest.raises(ValueError, match="Cannot interpret"):
            _item([(np.zeros((200, 2), dtype=np.float32),) * 2])


class TestAugmentationStaysPerSide:
    """
    Warping both sides of a pair identically moves both embeddings the same way
    and leaves the distance between them unchanged, which teaches the metric
    nothing. That rule has to survive the new sample kinds.
    """

    def test_identical_trajectories_are_warped_differently(self):
        trajectory = _trajectory()
        first, second, _ = _item([(trajectory, trajectory)], augment=True, seed=7)
        assert not torch.equal(first, second)

    def test_a_cross_modal_pair_is_warped_on_both_sides(self, tmp_path):
        pair = (_scan_path(tmp_path), _trajectory())
        plain = _item([pair])
        warped = _item([pair], augment=True, seed=7)
        assert not torch.equal(plain[0], warped[0]), "scan side was not augmented"
        assert not torch.equal(plain[1], warped[1]), "rendered side was not augmented"

    def test_augmentation_off_is_reproducible_for_trajectories(self):
        pair = (_trajectory(), _trajectory(phase=0.4))
        first = _item([pair])
        again = _item([pair])
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
