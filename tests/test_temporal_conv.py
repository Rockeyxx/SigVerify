"""Research arm: a Conv1d stage in front of the pen LSTM. Off by default; the shipped model has none."""
import pytest
import torch

from config.settings import TEMPORAL_INPUT_FEATURES
from src.models.siamese import TemporalSiamese


def _signatures(lengths):
    torch.manual_seed(0)
    seqs = [torch.randn(n, TEMPORAL_INPUT_FEATURES) for n in lengths]
    return seqs, torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True), torch.tensor(lengths)


class TestConvBeforeLstm:
    def test_the_shipped_model_has_no_conv_stage(self):
        model = TemporalSiamese()
        assert model.conv is None
        assert not any(key.startswith("conv.") for key in model.state_dict())

    def test_conv_features_are_added_to_what_the_lstm_already_sees(self):
        plain, arm = TemporalSiamese(), TemporalSiamese(conv_filters=32, conv_kernel=5)
        assert arm.lstm.input_size == plain.lstm.input_size + 32
        _, padded, lengths = _signatures([40, 25])
        assert arm.encode(padded, lengths).shape == (2, arm.embedding_dim)

    def test_a_checkpoint_carries_its_conv_stage(self):
        arm = TemporalSiamese(conv_filters=16, conv_kernel=7).eval()
        loaded = TemporalSiamese.from_state_dict(arm.state_dict())
        assert (loaded.conv.out_channels, loaded.conv.kernel_size) == (16, (7,))
        seqs, _, _ = _signatures([30])
        assert torch.allclose(loaded.encode(seqs[0][None]), arm.encode(seqs[0][None]))

    def test_a_checkpoint_without_one_loads_without_one(self):
        assert TemporalSiamese.from_state_dict(TemporalSiamese().state_dict()).conv is None

    def test_a_padded_batch_and_a_single_signature_give_the_same_embedding(self):
        """Training embeds padded batches and enrolment embeds one signature at a time. The deltas
        the model appends are not zero just past a signature's end, so without masking the conv
        would read them in a batch and zeros alone."""
        arm = TemporalSiamese(conv_filters=32, conv_kernel=5).eval()
        seqs, padded, lengths = _signatures([40, 25])
        together = arm.encode(padded, lengths)
        for i, seq in enumerate(seqs):
            assert torch.allclose(together[i], arm.encode(seq[None])[0], atol=1e-5)

    def test_an_even_kernel_is_refused(self):
        with pytest.raises(ValueError):
            TemporalSiamese(conv_filters=8, conv_kernel=4)

