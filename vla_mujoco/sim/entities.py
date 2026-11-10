"""Manipulable-entity model: categories, attributes and sampling.

Every object is a flat-topped primitive (box or upright cylinder) so the 4-cup VGC10 can seal on its top face.
Categories give objects a *name* ("cube", "slab", ...) independent of colour; the language generator uses
category, colour, relative size, location and ordering to refer to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PALETTE = {
    "red": (0.85, 0.15, 0.15),
    "green": (0.15, 0.70, 0.25),
    "blue": (0.15, 0.30, 0.85),
    "yellow": (0.90, 0.80, 0.10),
    "orange": (0.95, 0.50, 0.10),
    "purple": (0.55, 0.20, 0.75),
    "cyan": (0.10, 0.75, 0.80),
    "pink": (0.95, 0.45, 0.70),
}

# half-extent ranges (m). Cups span ~0.042 m from the tool axis, hence the lower bounds.
CATEGORIES = {
    "cube": dict(shape="box", f=(0.045, 0.065), z="same", mass=(0.15, 0.4)),
    "block": dict(shape="box", f=(0.045, 0.065), z=(0.035, 0.07), mass=(0.15, 0.4)),
    "cylinder": dict(shape="cylinder", f=(0.05, 0.065), z=(0.035, 0.07), mass=(0.15, 0.4)),
    "slab": dict(shape="box", f=(0.05, 0.065), z=(0.012, 0.02), mass=(0.05, 0.12)),
    "disc": dict(shape="cylinder", f=(0.052, 0.065), z=(0.012, 0.02), mass=(0.05, 0.12)),
}


@dataclass
class Entity:
    name: str  # pool body name, e.g. "obj3"
    idx: int
    category: str
    color: str
    shape: str  # "box" | "cylinder"
    dims: tuple  # (hx, hy, hz); for cylinders hx = hy = radius
    mass: float
    location: str = "table"  # "table" | "belt" (initial location)
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    belt_s: float = 0.0  # initial along-belt coordinate (belt items)
    table_y: float = 0.0
    attrs: dict = field(default_factory=dict)

    @property
    def hz(self) -> float:
        return float(self.dims[2])

    @property
    def footprint(self) -> float:
        """Radius of the circle enclosing the footprint (for spacing)."""
        return float(np.hypot(self.dims[0], self.dims[1]))

    @property
    def volume(self) -> float:
        hx, hy, hz = self.dims
        return float(np.pi * hx * hx * 2 * hz) if self.shape == "cylinder" else float(8 * hx * hy * hz)


def sample_entity(rng: np.random.Generator, idx: int, category: str, color: str, location: str) -> Entity:
    spec = CATEGORIES[category]
    if spec["shape"] == "cylinder":
        R = rng.uniform(*spec["f"])
        dims = (R, R, rng.uniform(*spec["z"]))
    else:
        hx, hy = rng.uniform(*spec["f"], 2)
        if spec["z"] == "same":
            hx = hy = hz = hx
        else:
            hz = rng.uniform(*spec["z"])
        dims = (float(hx), float(hy), float(hz))
    return Entity(
        name=f"obj{idx}",
        idx=idx,
        category=category,
        color=color,
        shape=spec["shape"],
        dims=tuple(float(v) for v in dims),
        mass=float(rng.uniform(*spec["mass"])),
        location=location,
    )
