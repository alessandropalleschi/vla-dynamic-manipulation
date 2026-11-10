# VGC10 v4.1: PyYAML in Docker

This overlay replaces only `scripts/build_uf850_vacuum.sh`.

The previous script invoked `python3 scripts/install_uf850_vacuum.py` **on the Ubuntu host**, which fails in Conda environments without PyYAML. The new script:

1. Adds `python3-yaml` to `docker/Dockerfile.uf850` if missing.
2. Builds the UF850 Docker image.
3. Runs `install_uf850_vacuum.py` using `/usr/bin/python3` **inside** the image with `/workspace` bind-mounted.
4. Compiles the C++ Gazebo plugin using the same image.

Run from the project root:

```bash
bash scripts/build_uf850_vacuum.sh
bash scripts/test_vgc10_fixture.sh
```

No `pip`/Conda packages are required on the Ubuntu host. A Docker rebuild is required for the dependency. Existing generated models and scene files are not overwritten by this overlay.
