"""V3 network definitions (PyTorch) and the per-layer shape table shared by every workstream.

Nets (V3 DECISIONS D1; exact widths of the MobileNet-style net are an OPEN decision, this is the proposal):
  * resnet20_a / resnet20_b: ResNet-20 for CIFAR-10 (He et al., CVPR 2016, Sec. 4.2): 3x3 conv 16,
    3 stages x 3 BasicBlocks (16, 32, 64 ch), stride 2 at the first block of stages 2 and 3, GAP, FC 64->10.
    Shortcut option A = identity, strided subsample + zero channel padding (parameter-free);
    option B = 1x1 stride-2 conv + BN projection where the shape changes.
  * mobilenet_s: proposed MobileNet-V1-style CIFAR-10 net: 3x3 conv stem, depthwise-separable blocks
    (dw3x3 + BN + ReLU, pw1x1 + BN + ReLU), three stride-2 stages, GAP, FC. ReLU (not ReLU6) so the V2
    requant contract (ReLU + clip) applies unchanged.

All convs are SAME-padded (pad = k // 2). BN is folded into conv weights/bias at quantization time.

`layer_table(net_name)` walks the model structure (no hooks) and returns one dict per hardware layer in
execution order, with the fields the cycle model, packer and DSE need.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)

# Proposed MobileNet-style config: (out_channels, stride) per depthwise-separable block.
MOBILENET_S_STEM = 32
MOBILENET_S_BLOCKS = ((64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (256, 2))


def conv_bn(ic: int, oc: int, k: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ic, oc, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(oc))


class BasicBlock(nn.Module):
    def __init__(self, ic: int, oc: int, stride: int, shortcut: str):
        super().__init__()
        self.ic, self.oc, self.stride, self.shortcut_kind = ic, oc, stride, shortcut
        self.c1 = conv_bn(ic, oc, 3, stride)
        self.c2 = conv_bn(oc, oc, 3, 1)
        self.proj = None
        if (stride != 1 or ic != oc) and shortcut == "B":
            self.proj = conv_bn(ic, oc, 1, stride)

    def skip(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.ic == self.oc:
            return x
        if self.proj is not None:
            return self.proj(x)
        # option A: subsample + zero-pad channels (pad equally on both sides, as in the original CIFAR code)
        x = x[:, :, :: self.stride, :: self.stride]
        p = self.oc - self.ic
        return F.pad(x, (0, 0, 0, 0, p // 2, p - p // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.c1(x))
        y = self.c2(y)
        return F.relu(y + self.skip(x))


class ResNet20(nn.Module):
    def __init__(self, shortcut: str = "B", num_classes: int = NUM_CLASSES):
        super().__init__()
        assert shortcut in ("A", "B")
        self.shortcut = shortcut
        self.stem = conv_bn(3, 16, 3)
        blocks, ic = [], 16
        for oc, stride in ((16, 1), (32, 2), (64, 2)):
            for i in range(3):
                blocks.append(BasicBlock(ic, oc, stride if i == 0 else 1, shortcut))
                ic = oc
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class DSBlock(nn.Module):
    def __init__(self, ic: int, oc: int, stride: int):
        super().__init__()
        self.dw = conv_bn(ic, ic, 3, stride, groups=ic)
        self.pw = conv_bn(ic, oc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pw(F.relu(self.dw(x))))


class MobileNetS(nn.Module):
    def __init__(self, stem: int = MOBILENET_S_STEM, blocks=MOBILENET_S_BLOCKS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = conv_bn(3, stem, 3)
        layers, ic = [], stem
        for oc, stride in blocks:
            layers.append(DSBlock(ic, oc, stride))
            ic = oc
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(ic, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(x))
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


NETS = ("resnet20_a", "resnet20_b", "mobilenet_s")


def build(net: str) -> nn.Module:
    if net == "resnet20_a":
        return ResNet20("A")
    if net == "resnet20_b":
        return ResNet20("B")
    if net == "mobilenet_s":
        return MobileNetS()
    raise ValueError(net)


# ---------------------------------------------------------------- shape table
def _conv_row(name, conv: nn.Conv2d, ih, iw, **extra) -> dict:
    kh, kw = conv.kernel_size
    s, p, g = conv.stride[0], conv.padding[0], conv.groups
    oh, ow = (ih + 2 * p - kh) // s + 1, (iw + 2 * p - kw) // s + 1
    ic, oc = conv.in_channels, conv.out_channels
    k = (ic // g) * kh * kw                      # reduction length per output
    row = dict(layer=name, kind="dw" if g == ic and g > 1 else "conv", ic=ic, oc=oc, ih=ih, iw=iw, oh=oh, ow=ow,
               kh=kh, kw=kw, stride=s, pad=p, groups=g, k=k, macs=oc * oh * ow * k, params=oc * k,
               relu=1, residual_from="", skip_kind="", gap=0)
    row.update(extra)
    return row


def layer_table(net: str) -> list[dict]:
    """Hardware layers in execution order. Fields: layer, kind (conv|dw|fc), ic, oc, ih, iw, oh, ow, kh, kw,
    stride, pad, groups, k, macs, params (weights only; BN folds into a per-OC bias), relu, residual_from
    (name of the layer whose INPUT is the skip source, '' if none), skip_kind (identity|A|B), gap (1 if a
    global average pool follows this layer)."""
    m = build(net)
    c, h, w = INPUT_SHAPE
    rows = [_conv_row("stem", m.stem[0], h, w)]
    h, w = rows[-1]["oh"], rows[-1]["ow"]
    prev = "stem"
    for i, b in enumerate(m.blocks):
        if isinstance(b, BasicBlock):
            r1 = _conv_row(f"b{i}_c1", b.c1[0], h, w)
            r2 = _conv_row(f"b{i}_c2", b.c2[0], r1["oh"], r1["ow"])
            same = b.stride == 1 and b.ic == b.oc
            kind = "identity" if same else b.shortcut_kind
            if b.proj is not None:
                rp = _conv_row(f"b{i}_proj", b.proj[0], h, w, relu=0)
                rows.append(rp)
                r2.update(residual_from=rp["layer"], skip_kind=kind)
            else:
                r2.update(residual_from=prev, skip_kind=kind)   # skip = input of block = output of `prev`
            rows += [r1, r2]
            prev = r2["layer"]
            h, w = r2["oh"], r2["ow"]
        else:
            rd = _conv_row(f"b{i}_dw", b.dw[0], h, w)
            rp = _conv_row(f"b{i}_pw", b.pw[0], rd["oh"], rd["ow"])
            rows += [rd, rp]
            prev = rp["layer"]
            h, w = rp["oh"], rp["ow"]
    rows[-1]["gap"] = 1
    fc = m.fc
    rows.append(dict(layer="fc", kind="fc", ic=fc.in_features, oc=fc.out_features, ih=1, iw=1, oh=1, ow=1,
                     kh=1, kw=1, stride=1, pad=0, groups=1, k=fc.in_features, macs=fc.in_features * fc.out_features,
                     params=fc.in_features * fc.out_features, relu=0, residual_from="", skip_kind="", gap=0))
    return rows


if __name__ == "__main__":
    for n in NETS:
        t = layer_table(n)
        print(n, "layers", len(t), "MACs", sum(r["macs"] for r in t), "weights", sum(r["params"] for r in t),
              "torch params", sum(p.numel() for p in build(n).parameters()))
