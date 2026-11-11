"""Native-resolution rendering, letterboxing and the grounding labels that must follow the letterbox."""

import numpy as np
import pytest

from vla_mujoco.sim.env import DynamicStackEnv, EnvConfig
from vla_mujoco.sim.imaging import letterbox, letterbox_geometry


def test_letterbox_geometry_and_padding():
    assert letterbox_geometry(480, 640, 224) == (168, 224, 28, 0)
    assert letterbox_geometry(1080, 1920, 640) == (360, 640, 140, 0)
    assert letterbox_geometry(100, 100, 100) == (100, 100, 0, 0)
    img = np.full((48, 64, 3), 200, np.uint8)
    out = letterbox(img, 32)
    assert out.shape == (32, 32, 3)
    assert out[:4].max() == 0 and out[-4:].max() == 0  # black bars (top/bottom for a landscape image)
    assert out[16, 16, 0] > 150  # content in the middle


def test_native_render_is_letterboxed_and_labels_follow():
    env = DynamicStackEnv(EnvConfig(seed=2, img_size=96, render_hw=(48, 64), cameras=("front", "wrist")))
    obs, info = env.reset(seed=2)
    assert env.render("front").shape == (48, 64, 3)  # native
    assert obs["images"]["front"].shape == (96, 96, 3)  # what the policy sees
    nh, nw, top, left = letterbox_geometry(48, 64, 96)
    uv, vis = info["priv"]["ref_uv"], info["priv"]["ref_vis"]
    assert np.all(uv[vis][:, 1] >= top / 96 - 1e-6) and np.all(uv[vis][:, 1] <= (top + nh) / 96 + 1e-6)
    assert vis.any()


def test_projection_matches_native_pixels_for_non_square_render():
    """A point at the camera's optical axis must land at the image centre, whatever the aspect ratio."""
    env = DynamicStackEnv(EnvConfig(seed=0, render=False, img_size=64, render_hw=(48, 64)))
    env.reset(seed=0)
    m, d = env.model, env.data
    cid = m.camera("overhead").id
    axis = -d.cam_xmat[cid].reshape(3, 3)[:, 2]  # camera looks along -z
    uv, vis = env.project(d.cam_xpos[cid] + 0.8 * axis, "overhead")
    assert vis and np.allclose(uv, [0.5, 0.5], atol=1e-6)
    # a point to the right of the axis moves u (not v)
    right = d.cam_xmat[cid].reshape(3, 3)[:, 0]
    uv2, _ = env.project(d.cam_xpos[cid] + 0.8 * axis + 0.1 * right, "overhead")
    assert uv2[0] > 0.5 and abs(uv2[1] - 0.5) < 1e-6


@pytest.mark.parametrize("patch,size,grid", [(8, 64, 8), (16, 64, 4), (16, 128, 8), (32, 128, 4)])
def test_cnn_patch_grid(patch, size, grid):
    torch = pytest.importorskip("torch")
    from vla_mujoco.vla.policy import VLAPolicy
    from vla_mujoco.vla.text import TextBackend, TextConfig

    be = TextBackend(TextConfig(kind="bow"))
    m = VLAPolicy(arch="mlp", text=be.cfg, patch=patch)
    imgs = [torch.rand(1, 1, 3, size, size) for _ in m.cameras]
    _, _, gl = m.features(imgs, torch.zeros(1, 1, 22), be.encode(["put the cube onto the disc"]), ground=True)
    assert gl[0].shape[-1] == grid * grid


def test_pretrained_vit_backbone_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vla_mujoco.vla.policy import VLAPolicy
    from vla_mujoco.vla.text import TextBackend, TextConfig

    be = TextBackend(TextConfig(kind="bow"))
    try:
        m = VLAPolicy(arch="mlp", text=be.cfg, vision="facebook/dinov2-small", vision_size=112)
        imgs = [torch.rand(1, 1, 3, 112, 112) for _ in m.cameras]
        _, _, gl = m.features(imgs, torch.zeros(1, 1, 22), be.encode(["put the cube onto the disc"]), ground=True)
    except Exception as e:  # no network / hub unavailable
        pytest.skip(f"pretrained ViT unavailable: {e}")
    assert gl[0].shape[-1] == 64  # 112 / 14 = 8 -> 8 x 8 patches
    assert not any("_bb" in k or "backbone" in k for k in m.state_dict())  # frozen backbone not checkpointed
