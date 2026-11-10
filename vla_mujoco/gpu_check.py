"""Verify the GPU is usable for training (CUDA) and for MuJoCo camera rendering (EGL).

    python -m vla_mujoco.gpu_check          (or: make gpu-check)

Exit code 0 if CUDA works; rendering on a software rasteriser is reported as a warning because it still
works (just slower).
"""

from __future__ import annotations

import os
import sys
import time


def check_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("[cuda]   torch not installed (pip install -e '.[train]')")
        return False
    print(f"[cuda]   torch {torch.__version__}  built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print(
            "[cuda]   NOT available. In a container: start it with --gpus=all and install a CUDA torch wheel."
        )
        return False
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"[cuda]   device {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
    x = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        x = x @ x
        x = x / x.norm()
    torch.cuda.synchronize()
    ms = (time.time() - t) * 1000
    print(f"[cuda]   10 matmuls 4096x4096: {ms:.0f} ms (bf16 ok: {torch.cuda.is_bf16_supported()})")
    return True


def check_render() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    xml = "<mujoco><worldbody><light pos='0 0 2'/><geom size='.1'/></worldbody></mujoco>"
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    try:
        r = mujoco.Renderer(m, 128, 128)
    except Exception as e:  # noqa: BLE001
        print(f"[render] FAILED to create a {os.environ['MUJOCO_GL']} context: {e}")
        return
    renderer = "unknown"
    try:
        from OpenGL import GL

        r._gl_context.make_current()
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
    except Exception:  # noqa: BLE001
        pass
    r.update_scene(d)
    r.render()
    t = time.time()
    for _ in range(50):
        r.update_scene(d)
        r.render()
    ms = (time.time() - t) / 50 * 1000
    soft = any(k in renderer.lower() for k in ("llvmpipe", "softpipe", "swrast"))
    print(
        f"[render] MUJOCO_GL={os.environ['MUJOCO_GL']}  GL_RENDERER={renderer}  {ms:.1f} ms/frame (128x128)"
    )
    if soft:
        print(
            "[render] WARNING: software rasteriser. For GPU rendering use an NVIDIA container with "
            "NVIDIA_DRIVER_CAPABILITIES=all (graphics) so libEGL_nvidia is injected."
        )


if __name__ == "__main__":
    ok = check_cuda()
    check_render()
    sys.exit(0 if ok else 1)
