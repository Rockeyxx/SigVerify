"""Tests for the data pipeline — preprocessing, quality gates, splitting, datasets."""

import os
import tempfile
import numpy as np
import cv2
import torch
import pytest


class TestPreprocessImage:

    def _make_test_image(self, h=300, w=400):
        """Create a temporary grayscale signature image."""
        img = np.ones((h, w), dtype=np.uint8) * 255
        # Draw a mock signature stroke
        cv2.line(img, (50, 150), (350, 150), 0, 3)
        cv2.line(img, (100, 100), (200, 200), 0, 2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    def test_output_shape(self):
        from src.data.dataset import preprocess_image
        path = self._make_test_image()
        result = preprocess_image(path)
        assert result.shape == (1, 256, 256)
        os.unlink(path)

    def test_output_normalized(self):
        from src.data.dataset import preprocess_image
        path = self._make_test_image()
        result = preprocess_image(path)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        os.unlink(path)

    def test_nonexistent_file_raises(self):
        from src.data.dataset import preprocess_image
        with pytest.raises(FileNotFoundError):
            preprocess_image("/nonexistent/path.png")


class TestQualityGate:

    def _make_sharp_image(self):
        img = np.ones((400, 400), dtype=np.uint8) * 255
        for i in range(50, 350, 5):
            cv2.line(img, (i, 100), (i + 30, 300), 0, 2)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    def _make_tiny_image(self):
        img = np.ones((30, 30), dtype=np.uint8) * 255
        cv2.circle(img, (15, 15), 5, 0, 1)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name

    def test_sharp_image_passes(self):
        from src.data.dataset import quality_gate
        path = self._make_sharp_image()
        passed, details = quality_gate(path)
        assert passed is True
        assert "sharpness" in details
        os.unlink(path)

    def test_tiny_image_fails(self):
        from src.data.dataset import quality_gate
        path = self._make_tiny_image()
        passed, details = quality_gate(path)
        assert passed is False
        os.unlink(path)


class TestWriterIndependentSplit:

    def test_no_overlap(self):
        from src.data.dataset import writer_independent_split
        ids = list(range(100))
        train, val, test = writer_independent_split(ids)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_all_ids_covered(self):
        from src.data.dataset import writer_independent_split
        ids = list(range(100))
        train, val, test = writer_independent_split(ids)
        assert set(train) | set(val) | set(test) == set(ids)

    def test_approximate_ratios(self):
        from src.data.dataset import writer_independent_split
        ids = list(range(100))
        train, val, test = writer_independent_split(ids, 0.7, 0.15, 0.15)
        assert 60 <= len(train) <= 80
        assert 10 <= len(val) <= 20
        assert 10 <= len(test) <= 20


class TestSpatialSignatureDataset:

    def test_len_and_getitem(self):
        from src.data.dataset import SpatialSignatureDataset
        imgs = []
        for _ in range(4):
            img = np.ones((100, 100), dtype=np.uint8) * 255
            cv2.line(img, (10, 50), (90, 50), 0, 2)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, img)
            imgs.append(tmp.name)

        pairs = [(imgs[0], imgs[1]), (imgs[2], imgs[3])]
        labels = [1, 0]
        ds = SpatialSignatureDataset(pairs, labels)
        assert len(ds) == 2

        img1, img2, label = ds[0]
        assert img1.shape == (1, 256, 256)
        assert img2.shape == (1, 256, 256)
        assert label.item() == 1.0

        for p in imgs:
            os.unlink(p)
