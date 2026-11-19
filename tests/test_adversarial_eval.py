"""The Threat 2 attack must stay inside its budget, or its curve means nothing."""
import torch

from tools.research.adversarial_eval import pgd


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3)

    def encode(self, x):
        return self.conv(x).mean(dim=(2, 3))


def test_pgd_stays_within_eps_and_pixel_range_and_moves_toward_reference():
    torch.manual_seed(0)
    model = TinyEncoder()
    x = torch.rand(3, 1, 12, 12)
    references = torch.randn(5, 4)
    eps = 4 / 255

    adv = pgd(model, x, references, eps)

    assert (adv - x).abs().max() <= eps + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1
    before = torch.cdist(model.encode(x), references).min(dim=1).values
    after = torch.cdist(model.encode(adv), references).min(dim=1).values
    assert (after < before).all()
    assert pgd(model, x, references, 0) is x
