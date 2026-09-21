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
