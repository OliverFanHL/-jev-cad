import numpy as np

from jev_tmcad.training.laya import _validation_macro_f1


def test_validation_uses_tokenized_choice_indices():
    # _tokenize stores integer option IDs, whereas source JSONL stores strings.
    items = [{"label": 0}, {"label": 1}, {"label": 2}]
    logits = np.asarray([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=float)
    assert _validation_macro_f1(items, logits, 3) == 1.0
