import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from w0.utils.evaluation_rng import evaluation_rng
from w0.trainer import Wan22Trainer


def draw():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_eval_rng_repeatability_and_training_sequence_restored():
    random.seed(71); np.random.seed(71); torch.manual_seed(71)
    expected = draw()
    random.seed(71); np.random.seed(71); torch.manual_seed(71)
    with evaluation_rng(42, 'cpu'): a = draw()
    with evaluation_rng(42, 'cpu'): b = draw()
    assert a == b
    assert draw() == expected


def test_exception_restores_rng_and_model_training_mode():
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    expected = draw()
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    model = SimpleNamespace(dit=SimpleNamespace(training=True), _inference_state={'large': True})
    trainer = object.__new__(Wan22Trainer)
    trainer.val_dataset = [1]
    trainer.cfg = {}; trainer.seed = 42
    trainer.model = model
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m, process_index=0, device='cpu')
    def fail():
        model.dit.training = False
        draw()
        raise RuntimeError('bad validation sample')
    trainer._evaluate_impl = fail
    trainer._set_dit_only_train_mode = lambda: setattr(model.dit, 'training', True)
    with pytest.raises(RuntimeError, match='bad validation'):
        trainer.evaluate()
    assert model.dit.training and model._inference_state is None
    assert draw() == expected
