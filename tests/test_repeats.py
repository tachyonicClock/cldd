import torch
from src.augment import FlatAutoAugment
from src.drift_detector import UnattachedLearner


class _Recorder:
    schema = None
    random_seed = 0
    x_dtype = torch.float32
    y_dtype = torch.long
    device = "cpu"

    def __init__(self):
        self.calls = []

    def batch_train(self, x, y):
        self.calls.append(x.clone())


def _learner(repeats, augment=None):
    rec = _Recorder()
    learner = UnattachedLearner.__new__(UnattachedLearner)
    learner.learner = rec
    learner.after_train = None
    learner.repeats = repeats
    learner.augment = augment
    return learner, rec


def test_first_step_uses_original_batch_then_augmented():
    x = torch.zeros(4, 3)
    learner, rec = _learner(3, augment=lambda x: x + 1)
    learner.batch_train(x, torch.zeros(4))
    assert len(rec.calls) == 3
    assert torch.equal(rec.calls[0], x)
    assert all(torch.equal(c, x + 1) for c in rec.calls[1:])


def test_one_repeat_trains_once():
    learner, rec = _learner(1)
    learner.batch_train(torch.zeros(2, 3), torch.zeros(2))
    assert len(rec.calls) == 1


def test_flat_autoaugment_keeps_shape_and_scale():
    torch.manual_seed(0)
    mean, std = torch.tensor([0.5, 0.4, 0.3]), torch.tensor([0.2, 0.2, 0.2])
    aug = FlatAutoAugment((3, 8, 8), mean, std)
    x = ((torch.rand(16, 3, 8, 8) - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)).reshape(16, -1)
    out = aug(x)
    assert out.shape == x.shape
    # Values stay within the normalized range of [0, 1] images
    lo = ((0 - mean) / std).min()
    hi = ((1 - mean) / std).max()
    assert out.min() >= lo - 1e-4 and out.max() <= hi + 1e-4
