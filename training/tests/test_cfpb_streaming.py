import pandas as pd

from training.cfpb import narrative_word_counts, read_narratives


def write_parquet(tmp_path):
    path = tmp_path / "c.parquet"
    pd.DataFrame({"narrative": ["a b", "c", "d e f", "g", "h  i"]}).to_parquet(path, index=False)
    return path


def test_read_narratives_keeps_requested_order_across_batches(tmp_path):
    path = write_parquet(tmp_path)
    assert read_narratives(path, [4, 0, 2], batch_size=2) == ["h  i", "a b", "d e f"]


def test_read_narratives_empty_selection(tmp_path):
    assert read_narratives(write_parquet(tmp_path), [], batch_size=2) == []


def test_narrative_word_counts_streams_all_rows(tmp_path):
    counts = narrative_word_counts(write_parquet(tmp_path), batch_size=2)
    assert counts.tolist() == [2, 1, 3, 1, 2]
