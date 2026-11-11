"""Compact VLA policy: vision tokens + language tokens + proprioception -> belief -> action chunks.

    cameras -> CNN -> patch tokens --+
    text    -> backend/encoder ------+--> fusion transformer (query token) --> per-step feature
    proprio ------------------------+                                          |
                                                    memory (arch): mlp | stack | gru   -> belief z_t
                                                                                          |
                                         heads: action chunk (regress | flow matching), vacuum (Bernoulli),
                                                auxiliary (target-relative position, belt speed, phase, step)

* Text: ``bow`` / ``scratch`` (trainable) or ``pretrained`` (frozen HF encoder, see ``vla/text.py``).
* Memory ablation (docs/next_steps.md): ``mlp`` no memory, ``stack`` last K frames, ``gru`` recurrent belief.
* Action head: ``regress`` (L1) or ``flow`` (conditional flow matching, Euler sampling).
* The auxiliary heads are training signals only; inference never sees privileged state.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .text import MAX_TOKENS, VOCAB_SIZE, TextBackend, TextConfig, hash_tokens

ACT_DIM = 5  # [vx, vy, vz, wz, vacuum]
CONT_DIM = 4  # continuous part (flow / regression); vacuum is a separate Bernoulli head
N_PHASES, N_STEPS = 7, 4
AUX_DIM = 7  # pick object relative to the tool (xyz), support relative to the tool (xyz), support speed along belt


def tokenize(text: str, max_len: int = MAX_TOKENS) -> torch.Tensor:  # backwards-compatible helper
    return hash_tokens(text, max_len)


def sincos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """(h*w, dim) 2-D sine/cosine position embedding."""
    q = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(q, device=device) / q))
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    out = [f(c.flatten()[:, None] * omega[None]) for c in (xs, ys) for f in (torch.sin, torch.cos)]
    return torch.cat(out, 1)


def time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[..., None] * 1000.0 * f
    return torch.cat([a.sin(), a.cos()], -1)


class ImageEncoder(nn.Module):
    """Small CNN -> grid of patch tokens (stride 16)."""

    def __init__(self, d_model=128):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(8, o), nn.GELU())

        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 96), blk(96, 128))
        self.proj = nn.Linear(128, d_model)
        self.d = d_model

    def forward(self, x):  # (B,3,H,W) in [0,1] -> (B,N,d)
        f = self.net(x - 0.5)
        B, C, H, W = f.shape
        tok = self.proj(f.flatten(2).transpose(1, 2))
        return tok + sincos_2d(H, W, self.d, x.device)


class TextEncoderModule(nn.Module):
    """Token embeddings (B,L,d) + padding mask for the chosen backend."""

    def __init__(self, cfg: TextConfig, d_model=128):
        super().__init__()
        self.kind = cfg.kind
        self.pos = nn.Parameter(torch.randn(cfg.max_len, d_model) * 0.02)
        if cfg.kind == "pretrained":
            self.adapter = nn.Sequential(nn.Linear(cfg.emb_dim, d_model), nn.LayerNorm(d_model))
        else:
            self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        if cfg.kind == "scratch":
            layer = nn.TransformerEncoderLayer(d_model, 4, 2 * d_model, 0.0, batch_first=True, norm_first=True)
            self.ctx = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, text: dict):
        mask = text["mask"]
        x = self.adapter(text["emb"]) if self.kind == "pretrained" else self.emb(text["ids"])
        x = x + self.pos[: x.shape[1]]
        if self.kind == "scratch":
            x = self.ctx(x, src_key_padding_mask=~mask)
        return x, ~mask  # True = padding


class FlowHead(nn.Module):
    """Conditional flow matching in action-chunk space: predicts the velocity field given the belief."""

    def __init__(self, cond_dim, out_dim, hidden=512, steps=8):
        super().__init__()
        self.out_dim, self.steps = out_dim, steps
        self.net = nn.Sequential(
            nn.Linear(cond_dim + out_dim + 64, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def velocity(self, cond, x, t):
        return self.net(torch.cat([cond, x, time_embedding(t)], -1))

    def loss(self, cond, target):  # target (..., out_dim) in [-1, 1]
        t = torch.rand(target.shape[:-1], device=target.device)
        eps = torch.randn_like(target)
        x_t = (1 - t[..., None]) * eps + t[..., None] * target
        return F.mse_loss(self.velocity(cond, x_t, t), target - eps)

    @torch.no_grad()
    def sample(self, cond):
        x = torch.randn(*cond.shape[:-1], self.out_dim, device=cond.device)
        for i in range(self.steps):
            t = torch.full(cond.shape[:-1], i / self.steps, device=cond.device)
            x = x + self.velocity(cond, x, t) / self.steps
        return x.clamp(-1, 1)


class VLAPolicy(nn.Module):
    def __init__(
        self,
        cameras=("front", "wrist"),
        proprio_dim=22,
        arch="gru",
        chunk=8,
        window=4,
        hidden=256,
        d_model=128,
        head="regress",
        flow_steps=8,
        text: TextConfig | dict | None = None,
    ):
        super().__init__()
        text = TextConfig(**text) if isinstance(text, dict) else (text or TextConfig())
        self.cameras, self.arch, self.chunk, self.window = tuple(cameras), arch, chunk, window
        self.head_kind, self.hidden, self.d = head, hidden, d_model
        self.text_cfg = text
        self.cfg = dict(
            cameras=list(cameras), proprio_dim=proprio_dim, arch=arch, chunk=chunk, window=window, hidden=hidden,
            d_model=d_model, head=head, flow_steps=flow_steps, text=vars(text).copy(),
        )  # fmt: skip
        self.img = ImageEncoder(d_model)
        self.cam_emb = nn.Parameter(torch.randn(len(cameras), 1, d_model) * 0.02)
        self.txt = TextEncoderModule(text, d_model)
        self.prop = nn.Linear(proprio_dim, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, 4, 2 * d_model, 0.0, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        mult = window if arch == "stack" else 1
        self.fuse = nn.Sequential(nn.Linear(d_model * mult, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True) if arch == "gru" else None
        if head == "flow":
            self.flow = FlowHead(hidden, chunk * CONT_DIM, steps=flow_steps)
        else:
            self.reg = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, chunk * CONT_DIM))
        self.vac = nn.Linear(hidden, chunk)
        self.aux = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, AUX_DIM))
        self.aux_phase = nn.Linear(hidden, N_PHASES)
        self.aux_step = nn.Linear(hidden, N_STEPS)
        # referring-expression grounding: per image patch, is this where the instructed object (pick / support) is?
        self.g_pick = nn.Linear(d_model, 1)
        self.g_sup = nn.Linear(d_model, 1)
        self.register_buffer("p_mean", torch.zeros(proprio_dim))
        self.register_buffer("p_std", torch.ones(proprio_dim))

    # ------------------------------------------------------------------ belief
    def features(self, imgs, proprio, text, h0=None, ground=False):
        """imgs: list over cameras of (B,T,3,H,W); proprio (B,T,P); text: dict from TextBackend (B,...)."""
        B, T = proprio.shape[:2]
        toks = []
        for i, im in enumerate(imgs):
            t = self.img(im.flatten(0, 1)) + self.cam_emb[i]
            toks.append(t)
        txt, pad = self.txt(text)  # (B,L,d)
        txt = txt.repeat_interleave(T, 0)
        pad = pad.repeat_interleave(T, 0)
        p = self.prop((proprio - self.p_mean) / self.p_std).flatten(0, 1).unsqueeze(1)
        seq = torch.cat([self.query.expand(B * T, -1, -1), p, *toks, txt], 1)
        n_pre = 2 + sum(t.shape[1] for t in toks)
        key_pad = torch.cat([torch.zeros(B * T, n_pre, dtype=torch.bool, device=seq.device), pad], 1)
        out = self.out_norm(self.fusion(seq, src_key_padding_mask=key_pad))  # (B*T, S, d)
        x = out[:, 0].reshape(B, T, -1)
        gl, off = None, 2
        if ground:  # logits over each camera's patches for [pick, support]: list of (B, T, 2, N)
            gl = []
            for t in toks:
                n = t.shape[1]
                patch = out[:, off : off + n]
                off += n
                gl.append(
                    torch.stack([self.g_pick(patch).squeeze(-1), self.g_sup(patch).squeeze(-1)], 1).view(B, T, 2, n)
                )
        if self.arch == "stack":  # concat the last `window` per-step features (first frame repeated)
            x = torch.cat([torch.cat([x[:, :1].repeat(1, k, 1), x], 1)[:, :T] for k in range(self.window)], -1)
        z = self.fuse(x)
        h = None
        if self.gru is not None:
            z, h = self.gru(z, h0)
        return (z, h, gl) if ground else (z, h)

    # ------------------------------------------------------------------- heads
    def loss_terms(self, z, target):
        """target (B,T,chunk,5): continuous dims in [-1,1], vacuum in {0,1}."""
        cont = target[..., :CONT_DIM].flatten(-2)
        if self.head_kind == "flow":
            act = self.flow.loss(z, cont)
        else:
            act = F.l1_loss(self.reg(z), cont)
        vac = F.binary_cross_entropy_with_logits(self.vac(z), target[..., CONT_DIM])
        return {"act": act, "vac": vac}

    def aux_terms(self, z, aux_t, phase_t, step_t):
        return {
            "aux": F.smooth_l1_loss(self.aux(z), aux_t),
            "phase": F.cross_entropy(self.aux_phase(z).flatten(0, 1), phase_t.flatten()),
            "step": F.cross_entropy(self.aux_step(z).flatten(0, 1), step_t.flatten()),
        }

    @staticmethod
    def ground_terms(gl, ref_uv, ref_vis):
        """Cross-entropy over patches against the patch containing the projected object centre (visible ones only).

        gl: list per camera of (B, T, 2, N); ref_uv (B, T, cams, 2, 2) in [0, 1); ref_vis (B, T, cams, 2).
        Returns (loss, accuracy)."""
        loss, acc, cnt = 0.0, 0.0, 0.0
        for c, lg in enumerate(gl):
            n = lg.shape[-1]
            g = int(round(n**0.5))
            uv = ref_uv[:, :, c]  # (B, T, 2, 2)
            col = (uv[..., 0] * g).floor().clamp(0, g - 1).long()
            row = (uv[..., 1] * g).floor().clamp(0, g - 1).long()
            idx = row * g + col
            vis = ref_vis[:, :, c].float()
            ce = F.cross_entropy(lg.flatten(0, 2), idx.flatten(), reduction="none").view_as(vis)
            loss = loss + (ce * vis).sum()
            acc = acc + ((lg.argmax(-1) == idx).float() * vis).sum()
            cnt = cnt + vis.sum()
        cnt = cnt.clamp(min=1.0)
        return loss / cnt, acc / cnt

    @torch.no_grad()
    def sample_chunk(self, z):
        """(...,hidden) -> (..., chunk, 5): actions with vacuum as a probability."""
        cont = self.flow.sample(z) if self.head_kind == "flow" else self.reg(z).clamp(-1, 1)
        cont = cont.view(*z.shape[:-1], self.chunk, CONT_DIM)
        return torch.cat([cont, torch.sigmoid(self.vac(z))[..., None]], -1)

    @classmethod
    def from_checkpoint(cls, ck: dict) -> VLAPolicy:
        cfg = dict(ck["cfg"])
        cfg["text"] = TextConfig(**cfg["text"])
        m = cls(**cfg)
        m.load_state_dict(ck["model"])
        return m


class PolicyRunner:
    """Closed-loop wrapper returning one 5-D action per control step.

    The belief path (encoders + fusion + recurrent state) runs every control step, so the hidden state sees every
    observation. A new action chunk is drawn from the head only every ``exec_horizon`` steps and executed
    open-loop (action chunking); with the flow head that is where multi-step sampling happens.
    """

    def __init__(self, policy: VLAPolicy, device="cpu", exec_horizon=4, text_backend: TextBackend | None = None):
        self.p, self.dev = policy.eval().to(device), device
        self.exec_horizon = max(1, min(int(exec_horizon), policy.chunk))
        self.backend = text_backend or TextBackend(policy.text_cfg)
        self.reset()

    def reset(self):
        self.h, self.hist, self.text, self.queue, self.replans = None, [], None, [], 0

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        if self.text is None:
            self.text = {
                k: (v.to(self.dev) if v is not None else None)
                for k, v in self.backend.encode([obs["instruction"]]).items()
            }
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
            z, _ = self.p.features(imgs, pr, self.text)
        else:
            z, self.h = self.p.features(imgs, pr, self.text, self.h)
        if not self.queue:  # re-plan: take the next `exec_horizon` actions of the predicted chunk
            chunk = self.p.sample_chunk(z[:, -1])[0, : self.exec_horizon].float().cpu().numpy()
            self.queue = list(chunk.astype(np.float32))
            self.replans += 1
        return self.queue.pop(0)
