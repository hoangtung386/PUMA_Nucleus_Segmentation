import torch

from models.decoders import MutualFeatureExchange, ParallelDecoders
from models.fpn_aggregator import FPNAggregator
from models.stage2_refiner import ResidualNucleiRefinerUNet


def test_stage2_refiner_output_shape():
    model = ResidualNucleiRefinerUNet(in_channels=21, out_classes=10)
    x = torch.randn(1, 21, 128, 128)
    out = model(x)
    assert out.shape == (1, 10, 128, 128)


def test_stage2_refiner_zero_init():
    model = ResidualNucleiRefinerUNet(in_channels=21, out_classes=10)
    assert model.outc.weight.abs().sum().item() == 0.0
    assert model.outc.bias.abs().sum().item() == 0.0


def test_fpn_aggregator_output():
    fpn = FPNAggregator(vit_dim=1024, cnn_dims=[40, 80, 160, 320], fpn_dim=256)
    vit_tokens = torch.randn(1, 257, 1024)
    cnn_features = [
        torch.randn(1, 40, 256, 256),
        torch.randn(1, 80, 128, 128),
        torch.randn(1, 160, 64, 64),
        torch.randn(1, 320, 32, 32),
    ]
    out = fpn(vit_tokens, cnn_features)
    assert set(out.keys()) == {"p1", "p2", "p3", "p4", "p5"}
    assert out["p1"].shape[1] == 256


def test_mutual_feature_exchange():
    mfe = MutualFeatureExchange(dim=256)
    f_t = torch.randn(1, 256, 32, 32)
    f_n = torch.randn(1, 256, 32, 32)
    ft_out, fn_out = mfe(f_t, f_n)
    assert ft_out.shape == f_t.shape
    assert fn_out.shape == f_n.shape


def test_parallel_decoders_output():
    decoders = ParallelDecoders(fpn_dim=256, num_tissue=5, num_nuclei=10)
    fpn_feats = {
        "p1": torch.randn(1, 256, 256, 256),
        "p2": torch.randn(1, 256, 128, 128),
        "p3": torch.randn(1, 256, 64, 64),
        "p4": torch.randn(1, 256, 32, 32),
        "p5": torch.randn(1, 256, 16, 16),
    }
    cp_prior = torch.randn(1, 2, 256, 256)
    tissue, np, nc, hv = decoders(fpn_feats, cp_prior)
    assert tissue.shape[1] == 5
    assert np.shape[1] == 1
    assert nc.shape[1] == 10
    assert hv.shape[1] == 2
