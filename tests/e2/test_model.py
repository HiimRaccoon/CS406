import copy
import pytest
import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.ops import StochasticDepth

class Spy(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.arange(1,769).float().reshape(1,768,1,1));self.calls=[]
    def forward(self,x):
        self.calls.append(x.detach().clone())
        return x.mean((1,2,3),keepdim=True)*self.weight

def setup():
    from src.e2.model import E2Model
    backbone=nn.Module();backbone.features=Spy();backbone.avgpool=nn.Identity();backbone.classifier=nn.Sequential(nn.LayerNorm([768,1,1]))
    m=E2Model(backbone,{'method':'linear_default','seed':1001})
    global_=torch.arange(2).float().add(1).reshape(2,1,1,1).expand(2,3,4,4).clone().requires_grad_()
    locals_=torch.arange(8).float().add(3).reshape(2,4,1,1,1).expand(2,4,3,4,4).clone().requires_grad_()
    return m,{'global':global_,'local':locals_}

def test_encoder_calls_global_then_batch_major_locals():
    m,v=setup();m(v)
    assert [len(x) for x in m.features.calls]==[2,8]
    assert torch.equal(m.features.calls[0],v['global'])
    assert torch.equal(m.features.calls[1],v['local'].flatten(0,1))

def test_fused_feature_is_global_then_local_mean():
    m,v=setup();g=m.encode_view(v['global']);locals_=m.encode_view(v['local'].flatten(0,1)).reshape(2,4,768)
    fused=m.encode(v)
    assert fused.shape==(2,1536)
    assert torch.equal(fused[:,:768],g) and torch.equal(fused[:,768:],locals_.mean(1))

def test_all_view_paths_backpropagate_to_one_encoder():
    m,v=setup();m(v).square().sum().backward()
    assert v['global'].grad is not None and torch.isfinite(v['global'].grad).all()
    assert v['local'].grad is not None and torch.isfinite(v['local'].grad).all()
    assert all(v['local'].grad[:,i].abs().sum()>0 for i in range(4))
    assert sum('features.weight'==k for k,_ in m.named_parameters())==1

def test_head_initialization_uses_fan_in_1536_and_restores_cpu_rng():
    from src.e2.model import E2Model
    b=convnext_tiny(weights=None,stochastic_depth_prob=.1);outer=torch.get_rng_state().clone()
    m=E2Model(b,{'method':'linear_default','seed':1001})
    assert torch.equal(outer,torch.get_rng_state())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1001);expected=nn.Linear(1536,124)
    assert torch.equal(m.classifier.weight,expected.weight) and torch.equal(m.classifier.bias,expected.bias)

def test_e2_architecture_parameter_count_matches_design_oracle(recipe):
    from src.e2.model import E2Model,build_optimizer
    b=convnext_tiny(weights=None,stochastic_depth_prob=.1);m=E2Model(b,{'method':'linear_default','seed':1001})
    assert sum(p.numel() for p in m.classifier.parameters())==190588
    assert sum(p.numel() for p in m.parameters())==28010716
    depth=[v.p for v in m.modules() if isinstance(v,StochasticDepth)]
    assert depth[0]==0 and depth[-1]==.1
    opt,groups=build_optimizer(m,{**recipe['optimizer'],'lr':1e-4})
    assert 'classifier.weight' in groups['decay'] and 'classifier.bias' in groups['no_decay']
    assert all(k in groups['no_decay'] for k,_ in m.named_parameters() if k.endswith('layer_scale'))
    assert len({id(p) for g in opt.param_groups for p in g['params']})==len(list(m.parameters()))


def test_digest_is_checked_in_full_before_loading(monkeypatch, tmp_path, recipe):
    import src.e2.model as module
    p = tmp_path / "convnext_tiny-983f1562.pth"
    p.write_bytes(b"synthetic, not pretrained")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Loaded unverified weights"))
    expected = recipe["model"]["pretrained"]["sha256"]
    with pytest.raises(ValueError):
        module.build_e2(p, expected, {**recipe["initialization"], "seed": 1001})
    with pytest.raises(ValueError):
        module.build_e2(p, "983f1562" + "0" * 56, {"seed": 1001})


def test_strict_verified_state_loading_is_preserved_at_external_edge(monkeypatch, tmp_path, recipe):
    import src.e2.model as module
    p = tmp_path / "convnext_tiny-983f1562.pth"
    backbone = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    torch.save(backbone.state_dict(), p)
    # Replace only file-integrity external edge. No alternate digest can be configured in production.
    monkeypatch.setattr(module, "sha256_file", lambda _: recipe["model"]["pretrained"]["sha256"])
    m, metadata = module.build_e2(p, recipe["model"]["pretrained"]["sha256"], {**recipe["initialization"], "seed": 1001})
    assert torch.equal(next(m.features.parameters()), next(backbone.features.parameters()))
    bad = backbone.state_dict()
    bad.pop(next(iter(bad)))
    torch.save(bad, p)
    with pytest.raises(RuntimeError):
        module.build_e2(p, recipe["model"]["pretrained"]["sha256"], {**recipe["initialization"], "seed": 1001})


def test_unknown_parameter_category_rejected(recipe):
    from src.e2.model import build_optimizer
    m = nn.Module()
    m.mystery = nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError):
        build_optimizer(m, {**recipe["optimizer"], "lr": 1e-4})
