import pandas as pd
import pytest

from training.fairness import flesch, gap, group_masks, per_group, quartile_labels, syllables
from training.metrics import accuracy


@pytest.mark.parametrize(
    "word,count", [("cat", 1), ("table", 2), ("cake", 1), ("readability", 5), ("XXXX", 1)]
)
def test_syllables(word, count):
    assert syllables(word) == count


def test_flesch_simple_text_scores_higher():
    simple = "The cat sat. The dog ran. I paid my bill."
    hard = "Notwithstanding contractual obligations, the institution misrepresented amortization."
    assert flesch(simple) > flesch(hard)
    assert flesch("") == 0.0


def test_quartile_labels_balanced_with_ties():
    labels = quartile_labels(pd.Series([1] * 8))
    assert labels.value_counts().to_dict() == {"Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2}


def test_group_masks_dimensions():
    df = pd.DataFrame(
        {
            "tags": ["Older American", "Servicemember", "", ""],
            "text": ["a b c.", "a b c d e f.", "a.", "a b."],
        }
    )
    masks = group_masks(df)
    assert set(masks) == {"tag", "length", "readability"}
    assert masks["tag"]["No tag"].tolist() == [False, False, True, True]


def test_per_group_and_gap():
    truth = ["a", "a", "b", "b", "a", "b"]
    pred = ["a", "a", "b", "b", "b", "a"]
    masks = {"g1": pd.Series([True] * 4 + [False] * 2), "g2": pd.Series([False] * 4 + [True] * 2)}
    out = per_group(truth, pred, masks, accuracy, min_n=2, n_boot=50)
    assert out["g1"]["value"] == 1.0 and out["g2"]["value"] == 0.0
    assert out["g1"]["n"] == 4
    assert gap(out) == 1.0
    small = per_group(truth, pred, masks, accuracy, min_n=5, n_boot=50)
    assert small["g2"]["value"] is None
    assert gap(small) is None
