import numpy as np

from jev_tmcad.evaluation.pilot_retrieval import _metrics_for_ranking


def test_retrieval_metrics_perfect_first_hit():
    labels = np.array(["a", "a", "b", "a"])
    ranking = np.array([1, 2, 3])
    m = _metrics_for_ranking(labels, 0, ranking, [1, 2])
    assert m["precision@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ap"] > 0.0
