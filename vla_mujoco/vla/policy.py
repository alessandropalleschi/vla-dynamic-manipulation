"""Compact VLA policy: multi-camera CNN + language + proprio -> (recurrent) action chunks.

Architectures (``arch``) for the memory ablation described in docs/next_steps.md:

* ``mlp``  - single frame, no memory
* ``stack``- fixed window of K past frames (stacked features)
* ``gru``  - recurrent belief (hidden state carried across control steps)

An auxiliary head regresses privileged target state (target xyz relative to the tool and
belt velocity) from the latent; it is only a training signal - inference needs no privileged input.
No pretrained weights are downloaded; swap ``ImageEncoder`` for a frozen ViT/DINO/SigLIP
backbone and ``TextEncoder`` for a language model to scale up.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import torch
from torch import nn

VOCAB_SIZE = 4096  # hashed word ids: any phrasing is representable (no OOV); swap for a pretrained encoder at scale
MAX_TOKENS = 40
ACT_DIM, VAC_IDX = 5, 4


def tokenize(text: str, max_len: int = MAX_TOKENS) -> torch.Tensor:
    ids = [1 + zlib.crc32(w.encode()) % (VOCAB_SIZE - 1) for w in re.findall(r"[a-z]+", text.lower())][:max_len]
    return torch.tensor(ids + [0] * (max_len - len(ids)), dtype=torch.long)


class ImageEncoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU())

        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128))
        self.kp = nn.Conv2d(128, 32, 1)  # spatial-softmax keypoints keep position info
        self.fc = nn.Linear(32 * 2 + 128, out)

    def forward(self, x):  # x: (B,3,H,W) float in [0,1]
        f = self.net(x - 0.5)
        B, C, H, W = f.shape
        k = torch.softmax(self.kp(f).flatten(2), -1).view(B, 32, H, W)
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W)
        kp = torch.cat([(k * xs).sum((2, 3)), (k * ys).sum((2, 3))], 1)
        return self.fc(torch.cat([kp, f.mean((2, 3))], 1))


class TextEncoder(nn.Module):
    def __init__(self, out=64, vocab=VOCAB_SIZE):
        super().__init__()
        self.emb = nn.Embedding(vocab, out, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(MAX_TOKENS, out) * 0.02)
        self.proj = nn.Sequential(nn.Linear(out, out), nn.GELU())

    def forward(self, tok):  # (B, MAX_TOKENS)
        e = (self.emb(tok) + self.pos) * (tok > 0).unsqueeze(-1)
        return self.proj(e.sum(1) / (tok > 0).sum(1, keepdim=True).clamp(min=1))


class VLAPolicy(nn.Module):
    def __init__(self, cameras=("front", "wrist"), proprio_dim=22, arch="gru", chunk=8, window=4, hidden=256):
        super().__init__()
        self.cameras, self.arch, self.chunk, self.window = tuple(cameras), arch, chunk, window
        self.img = ImageEncoder(128)
        self.cam_emb = nn.Parameter(torch.randn(len(cameras), 128) * 0.02)
        self.txt = TextEncoder(64)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 64), nn.GELU())
        feat = 128 * len(cameras) + 64 + 64
        mult = window if arch == "stack" else 1
        self.fuse = nn.Sequential(nn.Linear(feat * mult, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True) if arch == "gru" else None
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, chunk * ACT_DIM))
        self.aux = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 4))  # tgt-rel-tool xyz, belt vy
        self.register_buffer("p_mean", torch.zeros(proprio_dim))
        self.register_buffer("p_std", torch.ones(proprio_dim))
        self.hidden = hidden

    def step_features(self, imgs, proprio, tok):
        """imgs: list over cameras of (B,T,3,H,W); proprio (B,T,P); tok (B,12) -> (B,T,F)."""
        B, T = proprio.shape[:2]
        f = [self.img(im.flatten(0, 1)).view(B, T, -1) + self.cam_emb[i] for i, im in enumerate(imgs)]
        t = self.txt(tok).unsqueeze(1).expand(B, T, -1)
        p = self.prop((proprio - self.p_mean) / self.p_std)
        return torch.cat(f + [t, p], -1)

    def forward(self, imgs, proprio, tok, h0=None):
        x = self.step_features(imgs, proprio, tok)  # (B,T,F)
        if self.arch == "stack":  # concat the last `window` features
            x = torch.cat(
                [torch.cat([x[:, :1].repeat(1, k, 1), x], 1)[:, : x.shape[1]] for k in range(self.window)], -1
            )
        z = self.fuse(x)
        h = None
        if self.gru is not None:
            z, h = self.gru(z, h0)
        out = self.head(z).view(*z.shape[:2], self.chunk, ACT_DIM)
        return out, self.aux(z), h


class PolicyRunner:
    """Closed-loop wrapper returning one 5-D action per control step.

    The belief/feature path (encoders + recurrent state) runs every control step, so the hidden state sees
    every observation. A new action chunk is taken from the action head only every ``exec_horizon`` steps
    and its first ``exec_horizon`` actions run open-loop (action chunking). With a flow-matching head that
    is where multi-step sampling happens: ``control_hz / exec_horizon`` inferences per second.
    """

    def __init__(self, policy: VLAPolicy, device="cpu", exec_horizon=4):
        self.p, self.dev = policy.eval().to(device), device
        self.exec_horizon = max(1, min(int(exec_horizon), policy.chunk))
        self.reset()

    def reset(self):
        self.h, self.hist = None, []
        self.tok = None
        self.queue = []
        self.replans = 0

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        if self.tok is None:
            self.tok = tokenize(obs["instruction"]).unsqueeze(0).to(self.dev)
        imgs = [
            torch.from_numpy(obs["images"][c]).permute(2, 0, 1).float().div(255)[None, None].to(self.dev)
            for c in self.p.cameras
        ]
        pr = torch.from_numpy(obs["proprio"])[None, None].to(self.dev)
        if self.p.arch == "stack":
            self.hist.append((imgs, pr))
            self.hist = self.hist[-self.p.window :]
            hs = [self.hist[0]] * (self.p.window - len(self.hist)) + self.hist
            imgs = [torch.cat([h[0][i] for h in hs], 1) for i in range(len(self.p.cameras))]
            pr = torch.cat([h[1] for h in hs], 1)
            out, _, _ = self.p(imgs, pr, self.tok)
        else:
            out, _, self.h = self.p(imgs, pr, self.tok, self.h)
        if not self.queue:  # re-plan: take the next `exec_horizon` actions of the predicted chunk
            chunk = out[0, -1, : self.exec_horizon].float().cpu().numpy().copy()
            chunk[:, :4] = np.clip(chunk[:, :4], -1, 1)
            chunk[:, 4] = 1.0 / (1.0 + np.exp(-chunk[:, 4]))
            self.queue = list(chunk.astype(np.float32))
            self.replans += 1
        return self.queue.pop(0)
