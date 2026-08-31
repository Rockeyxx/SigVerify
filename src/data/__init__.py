# src/data/__init__.py
#
# Nothing in the repo, the tests, or the notebook imports from this package
# directly — every caller reaches for the submodule (`from src.data.dataset
# import ...`). The list below is therefore documentation of the public surface
# rather than the path anyone uses, so it is kept in step with what actually
# ships: it previously advertised the orphaned `generate_pairs` while omitting
# the functions every caller does use.
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
