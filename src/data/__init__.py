# Re-exports for package discovery; callers normally import from submodules.
from .dataset import (
    SpatialSignatureDataset,
    TemporalSignatureDataset,
    augment_image,
    augment_trajectory,
    collate_temporal_pairs,
    generate_skilled_pairs,
    pin_holdout,
    preprocess_image,
    quality_gate,
    trajectory_gate,
    writer_independent_split,
)
