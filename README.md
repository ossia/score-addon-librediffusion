# score-addon-librediffusion

Real-time diffusion for [ossia score](https://ossia.io) (and, through [Avendish](https://github.com/celtera/avendish),
Max/MSP, TouchDesigner, GStreamer, Godot, Python): SD1.5 / SD-turbo / SDXL / StreamV2V / FLUX.2-klein /
img2img-turbo, running on TensorRT through [librediffusion](https://github.com/jcelerier/librediffusion).

The object loads the librediffusion shared library at runtime through its C API, so it builds without CUDA
or TensorRT; it simply does nothing when the library is absent.

## The node

Every workflow goes through one frame pipeline: the inputs of a tick become a job (input frame, control map
or style image, embedding, interpolation factor), the job is rendered by the family's pipeline (SD /
klein / img2img-turbo), then optionally RIFE-interpolated. Two ways to run it:

* **Async off**: the frame is rendered on the render thread, one per tick.
* **Async on**: a worker thread owns the pipeline and renders jobs as they arrive; the render thread
  presents steady-clock-paced frames (and the RIFE sub-frames) so a slow model never stalls the host.
  `Async pacing` trades latency for continuity.

Both modes work for every workflow, including ControlNet, IP-Adapter and img2img-turbo.

Notable controls:

* `Interpolation exp` — RIFE frame interpolation (2^exp displayed frames per rendered frame). Needs
  `rife_ifnet_fp16.plan` in the engine folder or in its parent (one engine per GPU can serve many bundles);
  `train-lora.py --rife` builds it.
* `Manual mode` / `Trigger` — render only when `Trigger` fires, hold the last frame otherwise.
* `GPU` — CUDA device index. Every engine of the node is (re)loaded on that device.
* `Python cache`, `Build folder`, `Build options`, `Build` — the engine builder, below.

## Building engines from the node

The plugin embeds the exporter: a pinned [uv](https://github.com/astral-sh/uv) release plus
librediffusion's `train-lora.py` and its Python package. Pressing `Build` extracts them under
`Python cache` (default `C:\lrd` on Windows — keep it short, long venv paths break some wheels —
and `~/.cache/librediffusion` on Linux), then runs

    uv run --project <exporter> python train-lora.py <Build options> --output <Build folder>

with uv's cache, the managed Python 3.12 and the project venv all under `Python cache`, on the GPU
selected by `GPU`. The first build downloads the locked Python environment (torch, TensorRT, ... several
GB). Output goes to `<Build folder>/build.log`; the `Build status` outlet mirrors the state. When
`Build folder` is empty the bundle is built straight into the `Engines` folder, so it loads as soon as
it is complete.

`Build options` takes every `train-lora.py` argument except `--output`; see librediffusion's README for the
recipes (`--type sd15|sdxl|klein|img2img-turbo`, `--model`, `-l LORA`, `--controlnet`, `--ipadapter`,
`--rife`, ...).

The builder runs in its own process and belongs to the host process, not to the node: a build survives
the transport being stopped and restarted, and dies with the host.

## Building the addon

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

Requirements: a C++23 compiler, Boost >= 1.90 (`-DBOOST_ROOT=...`), network access for the first configure
(Avendish and uv are fetched). `-DLRD_EMBED_EXPORTER=OFF` builds without the embedded exporter.
Inside a score checkout the addon is picked up automatically from `src/addons`.

`presets/gen_presets.py` regenerates the score presets; port ids follow the order of `inputs_t`.
