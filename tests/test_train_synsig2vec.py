"""
Tests for the SynSig2Vec trainer's persistence.

Only `save_weights` is covered here. It is the piece a long run depends on: a
1000-epoch run is ~2.7 hours, one has already been lost at epoch 90, and a
checkpoint that is corrupt or absent is worth the same as no checkpoint at all.
"""
import pytest

# Closed experiment: skipped by default, run in CI (see pytest.ini).
pytestmark = pytest.mark.archived

import argparse
import os

import torch
from torch import nn

from tools.train_synsig2vec import save_weights


def _args():
    return argparse.Namespace(features="velocity", seed=42, split_seed=42,
                              epochs=1000)


def test_save_weights_round_trips(tmp_path):
    model = nn.Linear(4, 3)
    path = str(tmp_path / "w.pth")
    save_weights(model, _args(), 3, path, epoch=75)

    saved = torch.load(path, weights_only=True)
    assert saved["epoch"] == 75
    assert saved["split_seed"] == 42
    assert saved["channels"] == 3
    assert set(saved["state_dict"]) == set(model.state_dict())


def test_save_weights_leaves_no_temporary_behind(tmp_path):
    """The write goes via a temp file; a successful save must not leak it."""
    path = str(tmp_path / "w.pth")
    save_weights(nn.Linear(4, 3), _args(), 3, path, epoch=1)
    assert os.listdir(tmp_path) == ["w.pth"]


def test_save_weights_replaces_atomically(tmp_path):
    """
    An overwrite must never leave the previous checkpoint truncated.

    Every checkpoint overwrites the same path, so a non-atomic write would
    destroy the last good one on the way to writing a bad one.
    """
    path = str(tmp_path / "w.pth")
    save_weights(nn.Linear(4, 3), _args(), 3, path, epoch=25)
    first = os.path.getsize(path)

    save_weights(nn.Linear(64, 64), _args(), 3, path, epoch=50)
    assert torch.load(path, weights_only=True)["epoch"] == 50
    assert os.path.getsize(path) != first
    assert os.listdir(tmp_path) == ["w.pth"]


def test_save_weights_creates_missing_parent_directory(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "w.pth")
    save_weights(nn.Linear(4, 3), _args(), 3, path, epoch=1)
    assert os.path.exists(path)
