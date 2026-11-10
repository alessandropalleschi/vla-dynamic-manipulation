"""Text backends: turn instruction strings into token ids and (optionally) frozen pretrained token embeddings.

* ``bow``         hashed word ids, embedded and trained from scratch (baseline)
* ``scratch``     the same ids, contextualised by a small *trainable* transformer
* ``pretrained``  token embeddings from a *frozen* Hugging Face encoder (default all-MiniLM-L6-v2, 22M params),
                  projected by a trainable adapter. Embeddings are computed once per unique string and cached,
                  so the language model costs nothing per training step.

The frozen encoder never enters the policy's ``state_dict``; a checkpoint stores only the backend config and
the backend is rebuilt (and the model re-downloaded / loaded from cache) when evaluating.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

VOCAB_SIZE = 4096  # hashed word ids (no OOV)
MAX_TOKENS = 48
DEFAULT_PRETRAINED = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path.home() / ".cache" / "vla_mujoco_text"


def hash_tokens(text: str, max_len: int = MAX_TOKENS) -> torch.Tensor:
    ids = [1 + zlib.crc32(w.encode()) % (VOCAB_SIZE - 1) for w in re.findall(r"[a-z]+", text.lower())][:max_len]
    return torch.tensor(ids + [0] * (max_len - len(ids)), dtype=torch.long)


@dataclass
class TextConfig:
    kind: str = "bow"  # bow | scratch | pretrained
    name: str = DEFAULT_PRETRAINED  # only for kind == "pretrained"
    max_len: int = MAX_TOKENS
    emb_dim: int = 0  # filled in for pretrained


class TextBackend:
    """Encodes batches of strings; ``encode`` returns ``ids`` (N,L), ``mask`` (N,L) and ``emb`` (N,L,E) or None."""

    def __init__(self, cfg: TextConfig | None = None):
        self.cfg = cfg or TextConfig()
        self._lm = self._tok = None
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if self.cfg.kind == "pretrained":
            self._load_lm()  # also fixes emb_dim

    def _load_lm(self):
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.cfg.name)
        self._lm = AutoModel.from_pretrained(self.cfg.name).eval()
        for p in self._lm.parameters():
            p.requires_grad_(False)
        self.cfg.emb_dim = int(self._lm.config.hidden_size)

    def config(self) -> dict:
        return asdict(self.cfg)

    @torch.no_grad()
    def _embed_new(self, texts: list[str]):
        out = []
        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            x = self._tok(
                chunk, padding="max_length", truncation=True, max_length=self.cfg.max_len, return_tensors="pt"
            )
            h = self._lm(**x).last_hidden_state.half().numpy()
            for j, t in enumerate(chunk):
                self._cache[t] = (h[j], x["attention_mask"][j].numpy().astype(bool))
            out.extend(chunk)

    def encode(self, texts: list[str]) -> dict:
        ids = torch.stack([hash_tokens(t, self.cfg.max_len) for t in texts])
        if self.cfg.kind != "pretrained":
            return {"ids": ids, "mask": ids > 0, "emb": None}
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            self._embed_new(missing)
        emb = torch.from_numpy(np.stack([self._cache[t][0] for t in texts])).float()
        mask = torch.from_numpy(np.stack([self._cache[t][1] for t in texts]))
        return {"ids": ids, "mask": mask, "emb": emb}
