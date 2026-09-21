import torch

from models.dpcrn_noise import DPCRNDecoder, DPCRNEncoder

BATCH, FRAMES, BINS = 2, 401, 257


def test_encoder_downsamples_frequency_four_times_and_keeps_time():
    encoder = DPCRNEncoder()
    spectrum = torch.randn(BATCH, 2, BINS, FRAMES)
    bottleneck, skips = encoder(spectrum)
    assert bottleneck.shape == (BATCH, 128, 65, FRAMES)
    assert len(skips) == 5
    assert skips[0].shape == (BATCH, 32, 129, FRAMES)
    assert skips[1].shape == (BATCH, 32, 65, FRAMES)


def test_decoder_restores_full_frequency_resolution():
    encoder, decoder = DPCRNEncoder(), DPCRNDecoder()
    spectrum = torch.randn(BATCH, 2, BINS, FRAMES)
    bottleneck, skips = encoder(spectrum)
    mask = decoder(bottleneck, skips)
    assert mask.shape == (BATCH, 2, BINS, FRAMES)


def test_decoder_uses_skip_connections():
    """Perturbing only the first skip must change the mask."""
    encoder, decoder = DPCRNEncoder(), DPCRNDecoder()
    spectrum = torch.randn(BATCH, 2, BINS, FRAMES)
    bottleneck, skips = encoder(spectrum)
    baseline = decoder(bottleneck, skips)
    skips[0] = skips[0] + 1.0
    assert not torch.allclose(baseline, decoder(bottleneck, skips), atol=1e-4)


from models.dpcrn_noise import AttentionPooling, DPCRNNoiseClassifier, NoiseClassifierHead

SAMPLES = 64000


def test_head_returns_embedding_and_logits():
    head = NoiseClassifierHead()
    bottleneck = torch.randn(BATCH, 128, 65, FRAMES)
    magnitude = torch.randn(BATCH, BINS, FRAMES).abs()
    embedding, logits = head(bottleneck, magnitude)
    assert embedding.shape == (BATCH, 256)
    assert logits.shape == (BATCH, 36)


def test_forward_exposes_all_keys():
    model = DPCRNNoiseClassifier(classes_num=36)
    output = model(torch.randn(BATCH, SAMPLES))
    assert set(output) == {"noise", "logits", "mask", "embedding"}
    assert output["noise"].shape == (BATCH, SAMPLES)
    assert output["embedding"].shape == (BATCH, 256)


def test_embedding_depends_on_temporal_order():
    """Regression guard against reverting the whole head to spectral time-averaging.

    The superseded head was ``log1p(|noise_spectrum|).mean(dim=-1)``, which is
    invariant to time reversal up to STFT edge effects. This test only checks
    that the full head (strided frequency convs + BiGRU + pooling) is far more
    order-sensitive than that superseded head end-to-end; at random
    initialisation it cannot isolate which stage provides that sensitivity, so
    it does not by itself guard the attention-pooling mechanism — see
    ``test_attention_pooling_selects_the_frame_its_scores_favour`` and
    ``test_attention_pooling_is_not_a_mean_when_scores_are_sharp`` for that.
    """
    model = DPCRNNoiseClassifier(classes_num=36).eval()
    waveform = torch.randn(1, SAMPLES)
    reversed_waveform = waveform.flip(-1)
    window = torch.hann_window(model.n_fft)

    def superseded_head(signal):
        spectrum = torch.stft(
            signal, model.n_fft, model.hop_length, window=window,
            center=True, return_complex=True,
        ).abs()
        return torch.log1p(spectrum).mean(dim=-1)

    def sensitivity(first, second):
        return ((first - second).norm() / first.norm().clamp_min(1e-6)).item()

    with torch.no_grad():
        new = sensitivity(model(waveform)["embedding"], model(reversed_waveform)["embedding"])
        old = sensitivity(superseded_head(waveform), superseded_head(reversed_waveform))

    assert new > 10 * old


def test_attention_pooling_selects_the_frame_its_scores_favour():
    """Drive the scorer directly: the mechanism must follow its own weights."""
    pooling = AttentionPooling(4)
    sequence = torch.randn(1, 5, 4)
    with torch.no_grad():
        pooling.score.weight.zero_()
        pooling.score.bias.zero_()
        pooling.score.weight[0, 0] = 50.0
        sequence[0, :, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        pooled = pooling(sequence)
    assert torch.allclose(pooled, sequence[0, 2], atol=1e-3)


def test_attention_pooling_is_not_a_mean_when_scores_are_sharp():
    """Guards the mechanism itself: a mean-pool stand-in fails this."""
    pooling = AttentionPooling(4)
    sequence = torch.randn(1, 6, 4)
    with torch.no_grad():
        pooling.score.weight.zero_()
        pooling.score.bias.zero_()
        pooling.score.weight[0, 0] = 50.0
        pooled = pooling(sequence)
    assert not torch.allclose(pooled, sequence.mean(dim=1), atol=1e-2)
