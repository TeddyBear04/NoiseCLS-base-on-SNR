import numpy as np

from utils.reporting import label_by_snr_f1, markdown_table, metric_values

LABELS = ["a", "b"]


def test_metric_values_on_a_perfect_prediction():
    targets = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    values = metric_values(targets, probabilities.argmax(axis=1), probabilities)
    assert values["accuracy"] == 1.0
    assert values["macro_f1"] == 1.0


def test_label_by_snr_f1_splits_both_axes():
    targets = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 1])
    snrs = np.array([15.0, 15.0, 20.0, 20.0])
    table = label_by_snr_f1(targets, predictions, snrs, LABELS)
    assert set(table) == {"a", "b"}
    assert table["a"]["15"] == 1.0
    assert table["a"]["20"] == 0.0


def test_markdown_table_renders_a_header_rule():
    rendered = markdown_table(["x", "y"], [["1", "2"]])
    assert rendered.splitlines()[1] == "|---|---|"


def test_band_si_sdr_scores_the_requested_band_only():
    """A clean low band plus a corrupted high band must score worse at 4-8 kHz."""
    import torch

    from utils.reporting import band_si_sdr

    torch.manual_seed(0)
    time = torch.arange(16000) / 16000.0
    target = torch.sin(2 * torch.pi * 500 * time).unsqueeze(0)
    corrupted = target + 0.5 * torch.sin(2 * torch.pi * 6000 * time).unsqueeze(0)
    low = band_si_sdr(corrupted, target, 0.0, 4000.0)
    high = band_si_sdr(corrupted, target, 4000.0, 8000.0)
    assert high < low


def test_group_f1_averages_only_the_named_labels():
    import numpy as np

    from utils.reporting import group_f1

    labels = ["a", "b", "c"]
    targets = np.array([0, 1, 2, 0])
    predictions = np.array([0, 1, 0, 0])      # class "c" is always wrong
    assert group_f1(targets, predictions, labels, {"a", "b"}) > \
           group_f1(targets, predictions, labels, {"c"})
