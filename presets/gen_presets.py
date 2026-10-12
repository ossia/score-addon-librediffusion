#!/usr/bin/env python3
"""Regenerate the StreamDiffusion .scp presets.

Port ids come from LibreDiffusion.hpp's inputs_t order under Crousti/ProcessModelPortInit.hpp's
`int inlet = 0` counter (verified by the portids probe): the two texture inlets and the two plain
value inlets (Embedding, Trigger) consume 0..3, so the first saved control (Workflow) is 4.
The 2026-06-07 presets predate the Embedding and LoRA-scale ports and are off by one/two throughout.
Ports 26.. (GPU, the engine-builder controls and its Build impulse) are left at their defaults.

Every preset is checked against the bundle's real TensorRT profile (batch/resolution parsed from
/media/data2/lrd-engines/index.json's label, or the exporter's bundle.json) before it is written, so
a step count that the engine cannot run is an error here rather than a silent failure at load.
"""
import json
import os
import re
import shutil
import sys

OUT = "/home/jcelerier/Documents/ossia/score/packages/user/presets/StreamDiffusion"
UUID = "a202d577-f92e-4d47-b863-62be5c02084e"
INDEX = "/media/data2/lrd-engines/index.json"

# inputs_t order -> preset id
P = dict(workflow=4, prompt=5, negative=6, engines=7, seed=8, guidance=9, timesteps=10,
         resolution=11, cfg=12, add_noise=13, denoise_batch=14, manual=15, delta=16,
         feed_prev_in=17, feed_prev_out=18, cn_scale=19, ip_scale=20, lora_scale=21,
         klein_quality=22, rife_exp=23, async_=24, pacing=25, gpu=26, python_cache=27,
         build_folder=28, build_options=29, build=30)

# Workflow enum (LibreDiffusion.hpp)
(SD_T2I, SD_I2I, SD_T2I_CN, SD_I2I_CN, SD_T2I_IP, SD_I2I_IP, TURBO_T2I, TURBO_I2I,
 SDXL_T2I, SDXL_I2I, SDXL_T2I_CN, SDXL_I2I_CN, V2V_T2I, V2V_I2I,
 KLEIN_T2I, KLEIN_I2I, KLEIN_INPAINT, I2I_TURBO) = range(18)

CFG_NONE, CFG_SELF, CFG_FULL, CFG_INIT = 0, 1, 2, 3

# Start-index schedules. Mode matters, and the golden matrix cannot be used directly here: its cells
# exist to compare C++ against Python at the same settings, so it uses one schedule for both modes.
# Measured in score: a txt2img start of 32 leaves the latent under-noised and renders confetti, while
# an img2img start of 0 is ~pure noise (beta[0] ~= 1) and wipes the input frame. So txt2img starts at
# 0 and img2img mid-schedule -- except SDXS, which is distilled to jump from t999 and renders a
# featureless blur at 32 (A/B: t0 sharp, t20/t32 mush, with and without cfg-self).
# The img2img schedules stay inside the late band: every step before ~32 erodes the encoded frame,
# so a 4-step list starting at 16 renders a featureless gradient while the same 4 steps over 32..45
# keep the scene (measured on lcm-dreamshaper 4step; denoise-batch on/off makes no difference).
T2I_SCHED = {1: "0", 2: "0, 25", 3: "0, 16, 33", 4: "0, 12, 25, 37"}
I2I_SCHED = {1: "32", 2: "32, 45", 3: "32, 39, 45", 4: "32, 37, 42, 45"}
I2I_OVERRIDE = {"sdxs": {1: "0"}}

PROMPTS = {
    "t2i": ("a red fox in a snowy forest, oil painting, highly detailed", "low quality, blurry"),
    "i2i": ("a vivid impressionist oil painting, bold brushstrokes, vibrant saturated color",
            "low quality, blurry"),
    "cn_reinvent": ("creature from the woods, photorealistic", "low quality, blurry"),
    "cn_restyle": ("a vivid impressionist oil painting, bold brushstrokes, vibrant saturated color",
                   "low quality, blurry"),
    "ip": ("a portrait in the style of the reference image, highly detailed", "low quality, blurry"),
    "klein": ("turn this into a charcoal drawing", "blurry"),
    "i2it": ("a photo of a mountain landscape at golden hour", ""),
}


# ---------------------------------------------------------------- engine profiles

def parse_label(label):
    """'sdxl:...@b1-4_r1024sq_dynamic_o5' -> (max_batch, [(w_min,w_max),(h_min,h_max)])."""
    spec = label.rsplit("@", 1)[-1]
    b = re.search(r"b(\d+)-(\d+)", spec)
    max_batch = int(b.group(2)) if b else 1
    r = re.search(r"_r(\d+)sq", spec)
    if r:
        v = int(r.group(1))
        return max_batch, ((v, v), (v, v))
    r = re.search(r"_r(\d+)-(\d+)x(\d+)-(\d+)", spec)
    if r:
        a, b2, c, d = (int(x) for x in r.groups())
        return max_batch, ((a, b2), (c, d))
    r = re.search(r"_r(\d+)-(\d+)", spec)
    if r:
        a, b2 = int(r.group(1)), int(r.group(2))
        return max_batch, ((a, b2), (a, b2))
    return max_batch, None


def load_profiles():
    prof = {}
    if os.path.isfile(INDEX):
        for h, e in json.load(open(INDEX))["entries"].items():
            mb, res = parse_label(e.get("label", ""))
            prof[e.get("dir", "/media/data2/lrd-engines/" + h)] = (mb, res, e.get("label", ""))
    return prof


PROFILES = load_profiles()


def bundle_profile(d):
    """Profile for a directory: index.json first, then the exporter's bundle.json."""
    if d in PROFILES:
        return PROFILES[d]
    bj = os.path.join(d, "bundle.json")
    if os.path.isfile(bj):
        try:
            m = json.load(open(bj))
        except Exception:
            return None
        mb = (m.get("batch") or {}).get("max", 1)
        r = m.get("resolution") or {}
        if "min" in r and "max" in r:
            res = ((r["min"], r["max"]), (r["min"], r["max"]))
        elif "width" in r:
            res = ((r["width"], r["width"]), (r["height"], r["height"]))
        else:
            res = None
        return mb, res, m.get("base_model", "")
    return None


# ---------------------------------------------------------------- preset building

problems = []


def preset(name, engines, workflow, *, steps=1, res=(512, 512), mode="t2i", sched=None, timesteps=None,
           cfg=None, guidance=None, add_noise=None, denoise_batch=None, cn_scale=1.0,
           ip_scale=0.7, lora_scale=1.0, klein_quality=0, rife_exp=0, is_async=False, pacing=0,
           prompt_key=None, seed=42, delta=1.0, check=True):
    if not os.path.isdir(engines):
        problems.append(f"{name}: engine dir missing: {engines}")
        return None

    if check:
        p = bundle_profile(engines)
        if p is None:
            problems.append(f"{name}: no profile metadata for {engines} (unchecked)")
        else:
            max_batch, rng, _ = p
            # denoise-batch folds the N steps into ONE UNet call of batch N; sequential stays at 1.
            need = steps if denoise_batch else 1
            if need > max_batch:
                problems.append(
                    f"{name}: needs batch {need} but engine profile maxes at {max_batch}")
                return None
            if rng:
                (wlo, whi), (hlo, hhi) = rng
                if not (wlo <= res[0] <= whi and hlo <= res[1] <= hhi):
                    problems.append(
                        f"{name}: {res[0]}x{res[1]} outside engine profile "
                        f"{wlo}-{whi} x {hlo}-{hhi}")
                    return None

    is_i2i = mode in ("i2i", "cn_restyle", "ip_i2i")
    if timesteps is None:
        table = I2I_SCHED if is_i2i else T2I_SCHED
        if is_i2i and sched in I2I_OVERRIDE and steps in I2I_OVERRIDE[sched]:
            timesteps = I2I_OVERRIDE[sched][steps]
        else:
            timesteps = table[steps]
    if cfg is None:
        cfg = CFG_NONE
    if guidance is None:
        guidance = 1.0
    if add_noise is None:
        add_noise = not is_i2i
    if denoise_batch is None:
        denoise_batch = False
    pos, neg = PROMPTS[prompt_key or mode]

    vals = {
        P["workflow"]: {"Int": workflow},
        P["prompt"]: {"String": pos},
        P["negative"]: {"String": neg},
        P["engines"]: {"String": engines},
        P["seed"]: {"Int": seed},
        P["guidance"]: {"Float": guidance},
        P["timesteps"]: {"String": timesteps},
        P["resolution"]: {"Vec2f": [float(res[0]), float(res[1])]},
        P["cfg"]: {"Int": cfg},
        P["add_noise"]: {"Bool": bool(add_noise)},
        P["denoise_batch"]: {"Bool": bool(denoise_batch)},
        P["manual"]: {"Bool": False},
        P["delta"]: {"Float": delta},
        P["feed_prev_in"]: {"Float": 0.0},
        P["feed_prev_out"]: {"Float": 0.0},
        P["cn_scale"]: {"Float": cn_scale},
        P["ip_scale"]: {"Float": ip_scale},
        P["lora_scale"]: {"Float": lora_scale},
        P["klein_quality"]: {"Int": klein_quality},
        P["rife_exp"]: {"Int": rife_exp},
        P["async_"]: {"Bool": bool(is_async)},
        P["pacing"]: {"Int": pacing},
    }
    doc = {"Key": {"Uuid": UUID, "Effect": ""}, "Name": name,
           "Preset": [[i, vals[i]] for i in sorted(vals)]}
    path = os.path.join(OUT, name + ".scp")
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    return name


E = "/media/data2/lrd-engines"


def build():
    made = []

    # ---- SD1.5 family: 768-dim CLIP, pad 49407, cfg + denoise-batch honoured from the ports.
    sd15 = [
        ("SD15 SDXS 512",            f"{E}/44549d33ab82fefb", 1, (512, 512), "sdxs"),
        ("SD15 Hyper-SD 1step",      f"{E}/32fde15b1bc3abf8", 1, (512, 512), "hyper-sd15"),
        ("SD15 Hyper-SD 2step",      f"{E}/a93990bd296e3fe9", 2, (512, 512), "hyper-sd15"),
        ("SD15 Hyper-SD 4step",      f"{E}/c049cd0e1923be58", 4, (512, 512), "hyper-sd15"),
        ("SD15 LCM Dreamshaper 2step", f"{E}/611a46b8d791e296", 2, (512, 512), "lcm"),
        ("SD15 LCM Dreamshaper 4step", f"{E}/5df487d86a94d462", 4, (512, 512), "lcm"),
        ("SD21 base 1step",  "/media/data1/lrd-benchmark/engines/sd21-base-1step", 1, (512, 512), "sd21"),
        ("SD21 base 4step",  "/media/data1/lrd-benchmark/engines/sd21-base-4step", 4, (512, 512), "sd21"),
    ]
    for name, eng, steps, res, sch in sd15:
        db = steps > 1          # 1.45x at 2 steps / 1.6x at 4, and it costs no extra VRAM
        made += [preset(f"{name} txt2img", eng, SD_T2I, steps=steps, res=res, mode="t2i",
                        sched=sch, denoise_batch=db),
                 preset(f"{name} img2img", eng, SD_I2I, steps=steps, res=res, mode="i2i",
                        sched=sch, denoise_batch=db)]

    # ---- SD-Turbo: 1024-dim CLIP, pad 0, single-step by construction (the node forces both).
    for name, eng, res in [
        ("SD-Turbo 512",      f"{E}/abb77149636ece27", (512, 512)),
        ("SD-Turbo 512x768",  f"{E}/e8f27dfc7f88d76f", (512, 768)),
        ("SD-Turbo 768x512",  f"{E}/1550845aa0f77668", (768, 512)),
        ("SD-Turbo 768",      f"{E}/fcf0c5cde3f4ee27", (768, 768)),
    ]:
        made += [preset(f"{name} txt2img", eng, TURBO_T2I, steps=1, res=res, mode="t2i",
                        sched="sd-turbo"),
                 preset(f"{name} img2img", eng, TURBO_I2I, steps=1, res=res, mode="i2i",
                        sched="sd-turbo")]

    # ---- SDXL family: 2048-dim CLIP + clip2, pooled embeds. The node pins cfg-none/guidance 0 on
    # this path, so the img2img presets cannot use the cfg-self restyle recipe.
    sdxl = [
        ("SDXL Turbo 512",           f"{E}/b34b954f2133d4f0", 1, (512, 512), "sdxl-turbo"),
        ("SDXL Turbo 1024",          f"{E}/b7ee298aa818aa51", 1, (1024, 1024), "sdxl-turbo"),
        ("SDXL Hyper-SD 4step",      f"{E}/847eb1390454cf93", 4, (1024, 1024), "sdxl-4step"),
        ("SDXL Lightning 4step",     f"{E}/c8d002f4bf80148a", 4, (1024, 1024), "sdxl-4step"),
        ("SDXL LCM-LoRA 2step",      f"{E}/d652fc8897e4587c", 2, (1024, 1024), "sdxl-lcm"),
        ("SDXL LCM-LoRA 2step 512",  f"{E}/f5a48e2ed23d671c", 2, (512, 512), "sdxl-lcm"),
        ("SDXL LCM-LoRA 3step",      f"{E}/0f1da52837e59afa", 3, (1024, 1024), "sdxl-lcm"),
        ("SDXL LCM-LoRA 4step",      f"{E}/40ddf8ec5cec3718", 4, (1024, 1024), "sdxl-lcm"),
        ("SDXL Segmind VegaRT 2step", "/media/data1/lrd-engines/vega", 2, (1024, 1024), "vega"),
    ]
    for name, eng, steps, res, sch in sdxl:
        slow = res[0] >= 1024      # ~9-17 fps at 1024: diffuse off the render thread
        db = steps > 1             # the form the multi-step SDXL goldens were produced with
        made += [preset(f"{name} txt2img", eng, SDXL_T2I, steps=steps, res=res, mode="t2i",
                        sched=sch, denoise_batch=db, is_async=slow),
                 preset(f"{name} img2img", eng, SDXL_I2I, steps=steps, res=res, mode="i2i",
                        sched=sch, denoise_batch=db, is_async=slow)]

    # ---- 1-2 step SDXL UNet replacements (train-lora.py --unet, bench/build_fast_models.sh). One
    # 1024 b1-2 bundle each, so every one runs at 1 and at 2 steps. NitroFusion's shifted timestep is
    # in its bundle.json (unet_timestep_scale), not in the preset. DMD2's 1-step UNet only ever saw
    # pure noise at t=399: it is txt2img-only (img2img input comes out as a soft blur), from index 30.
    F = "/media/data2/lrd-engines-fast"
    made += [preset("SDXL DMD2 1step-UNet txt2img", f"{F}/dmd2-sdxl-1step", SDXL_T2I, steps=1,
                    res=(1024, 1024), mode="t2i", timesteps="30", is_async=True)]
    for name, eng in [
        ("SDXL DMD2 4step-UNet", f"{F}/dmd2-sdxl-4step"),
        ("SDXL NitroSD Realism", f"{F}/nitrosd-realism"),
        ("SDXL NitroSD Vibrant", f"{F}/nitrosd-vibrant"),
    ]:
        for steps in (1, 2):
            made += [preset(f"{name} {steps}step txt2img", eng, SDXL_T2I, steps=steps,
                            res=(1024, 1024), mode="t2i", denoise_batch=steps > 1, is_async=True),
                     preset(f"{name} {steps}step img2img", eng, SDXL_I2I, steps=steps,
                            res=(1024, 1024), mode="i2i", denoise_batch=steps > 1, is_async=True)]

    # ---- DMD2 at its native 4 steps: only txt2img on the trained schedule (999/759/499/259) is sharp;
    # img2img on more than 2 late steps goes soft, so its video presets stay at 2 steps. Needs the
    # batch-4 bundle (denoise-batch runs the 4 steps as one batch-4 UNet call).
    made += [preset("SDXL DMD2 4step-UNet 4step txt2img", f"{F}/dmd2-sdxl-4step-b4", SDXL_T2I,
                    steps=4, res=(1024, 1024), mode="t2i", timesteps="0, 12, 25, 37",
                    denoise_batch=True, is_async=True)]

    # ---- RIFE between diffused frames (the producer interpolates prev -> cur; async only). The
    # factors put the displayed rate at ~20-45 fps on a 4090 at 1024 (sd_video_harness --rife-exp):
    # turbo 1 step 12.6 diffused / 25 displayed (x2); turbo 2 steps 7.8 / 31 (x4); DMD2 2 steps
    # 8.0 / 32 (x4); DMD2 4 steps 4.8 / 19 (x4). The engine is found next to the bundle or in its
    # parent folder (one rife_ifnet_fp16.plan per GPU in F).
    for name, eng, steps, exp, t2i_ts, i2i_ts in [
        ("SDXL Turbo 1step RIFE x2", f"{F}/sdxl-turbo", 1, 1, None, None),
        ("SDXL Turbo 2step RIFE x4", f"{F}/sdxl-turbo", 2, 2, None, None),
        ("SDXL DMD2 4step-UNet 2step RIFE x4", f"{F}/dmd2-sdxl-4step", 2, 2, None, None),
        ("SDXL DMD2 4step-UNet 4step RIFE x4", f"{F}/dmd2-sdxl-4step-b4", 4, 2, "0, 12, 25, 37", False),
    ]:
        made += [preset(f"{name} txt2img", eng, SDXL_T2I, steps=steps, res=(1024, 1024),
                        mode="t2i", timesteps=t2i_ts, denoise_batch=steps > 1, rife_exp=exp,
                        is_async=True)]
        if i2i_ts is not False:
            made += [preset(f"{name} img2img", eng, SDXL_I2I, steps=steps, res=(1024, 1024),
                            mode="i2i", timesteps=i2i_ts, denoise_batch=steps > 1, rife_exp=exp,
                            is_async=True)]

    # ---- ControlNet. Preprocessing is EXTERNAL: feed an already-extracted canny/depth/pose/softedge
    # map into "Control / Style". txt2img+CN reinvents from the structure, img2img+CN restyles the
    # actual footage.
    cn_sd15 = [
        ("SD15 ControlNet Canny",    "/media/data2/cn-sd15-test/engines"),
        ("SD15 ControlNet Depth",    "/media/data2/cn-sd15-depth/engines"),
        ("SD15 ControlNet SoftEdge", "/media/data2/cn-sd15-softedge/engines"),
        ("SD15 ControlNet OpenPose", "/media/data2/cn-sd15-openpose/engines"),
        ("SDXS ControlNet Sketch",   "/media/data2/cn-sdxs-sketch/engines"),
        ("SDXS ControlNet Sketch (portable)", "/media/data2/cn-sdxs-sketch-portable/engines"),
    ]
    for name, eng in cn_sd15:
        sch = "sdxs" if name.startswith("SDXS") else "controlnet-sd15"
        made += [preset(f"{name} reinvent", eng, SD_T2I_CN, steps=1, res=(512, 512), sched=sch,
                        mode="t2i", prompt_key="cn_reinvent", check=False),
                 preset(f"{name} restyle", eng, SD_I2I_CN, steps=1, res=(512, 512), sched=sch,
                        mode="i2i", prompt_key="cn_restyle", check=False)]
    for name, eng in [("SDXL ControlNet Canny", "/media/data2/cn-sdxl-test/engines"),
                      ("SDXL ControlNet OpenPose", "/media/data2/cn-sdxl-openpose/engines")]:
        made += [preset(f"{name} reinvent", eng, SDXL_T2I_CN, steps=1, res=(1024, 1024),
                        mode="t2i", sched="sdxl-turbo", prompt_key="cn_reinvent",
                        is_async=False, check=False),
                 preset(f"{name} restyle", eng, SDXL_I2I_CN, steps=1, res=(1024, 1024),
                        mode="i2i", sched="sdxl-turbo", prompt_key="cn_restyle",
                        is_async=False, check=False)]

    # ---- IP-Adapter: the style image goes on the "Control / Style" inlet and is encoded on-device.
    made += [preset("SD15 IP-Adapter txt2img", "/media/data2/ip-sd15-test/engines", SD_T2I_IP,
                    steps=1, res=(512, 512), mode="t2i", sched="ipadapter-sd15",
                    prompt_key="ip", check=False),
             preset("SD15 IP-Adapter img2img", "/media/data2/ip-sd15-test/engines", SD_I2I_IP,
                    steps=1, res=(512, 512), mode="i2i", sched="ipadapter-sd15",
                    prompt_key="ip", check=False)]

    # ---- FLUX.2-klein. Timesteps here is a FlowMatch sigma list, not SD indices; the SD-style
    # default leaves the model on its native 2-step schedule, which is what the demos used.
    K = "/media/data2/flux-trt/klein-bundle-0620"
    for label, quality in (("Quality bf16", 0), ("Speed fp8", 1)):
        made += [preset(f"FLUX2 Klein {label} txt2img", K, KLEIN_T2I, steps=1, res=(320, 576),
                        mode="t2i", timesteps="15, 25", prompt_key="klein", klein_quality=quality,
                        rife_exp=2, is_async=True, pacing=0, check=False),
                 preset(f"FLUX2 Klein {label} img2img", K, KLEIN_I2I, steps=1, res=(320, 576),
                        mode="i2i", timesteps="15, 25", prompt_key="klein", klein_quality=quality,
                        rife_exp=2, is_async=True, pacing=0, check=False)]
    made += [preset("FLUX2 Klein inpaint", K, KLEIN_INPAINT, steps=1, res=(320, 576), mode="i2i",
                    timesteps="15, 25", prompt_key="klein", klein_quality=0, rife_exp=0,
                    is_async=True, check=False),
             preset("FLUX2 Klein landscape", "/media/data1/klein-landscape", KLEIN_I2I, steps=1,
                    res=(576, 320), mode="i2i", timesteps="15, 25", prompt_key="klein",
                    klein_quality=1, rife_exp=2, is_async=True, check=False)]

    # ---- klein + the SANA-Sprint LoRA (fused at export): good at 1 step, best at 2. "1.0" is the
    # one-step sigma list; the SD-style default keeps the native 2-step schedule.
    KS = "/media/data2/lrd-engines-fast/klein-sprint"
    for steps, ts in ((1, "1.0"), (2, "15, 25")):
        made += [preset(f"FLUX2 Klein Sprint {steps}step txt2img", KS, KLEIN_T2I, steps=1,
                        res=(320, 576), mode="t2i", timesteps=ts, prompt_key="klein",
                        klein_quality=1, rife_exp=2, is_async=True, check=False),
                 preset(f"FLUX2 Klein Sprint {steps}step img2img", KS, KLEIN_I2I, steps=1,
                        res=(320, 576), mode="i2i", timesteps=ts, prompt_key="klein",
                        klein_quality=1, rife_exp=2, is_async=True, check=False)]

    # ---- img2img-turbo (skip-VAE pix2pix-turbo): geometry and step count come from the engine.
    for name, eng in [("img2img-turbo edge2image", "/media/data1/img2img-turbo/bundle-traintest"),
                      ("img2img-turbo sketch2image", "/media/data1/img2img-turbo/bundle-sketch")]:
        made += [preset(name, eng, I2I_TURBO, steps=1, res=(512, 512), mode="i2i",
                        timesteps="0", prompt_key="i2it", check=False)]

    # ---- runtime-LoRA demo: engine exported with --lora PATH:runtime, so LoRA scale is live.
    made += [preset("SDXL crayon runtime-LoRA txt2img", "/media/data1/lora-test/crayon-rt",
                    SDXL_T2I, steps=1, res=(1024, 1024), mode="t2i", sched="sdxl-turbo",
                    lora_scale=1.0, is_async=True)]

    return [m for m in made if m]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate the StreamDiffusion .scp presets.")
    ap.add_argument("--out", default=OUT, help=f"preset folder (default {OUT})")
    ap.add_argument("--only", metavar="REGEX",
                    help="write only the presets whose name matches, leaving every other file in "
                         "--out alone (default: wipe --out and write them all)")
    args = ap.parse_args()
    OUT = args.out
    if not os.path.isdir(OUT):
        sys.exit(f"preset dir not found: {OUT}")
    if args.only:
        only = re.compile(args.only)
        _preset = preset
        def preset(name, *a, **kw):  # noqa: F811 -- build() looks it up at call time
            return _preset(name, *a, **kw) if only.search(name) else None
    else:
        backup = OUT + ".bak-2026-07-26"
        if not os.path.exists(backup):
            shutil.copytree(OUT, backup)
            print(f"backed up existing presets -> {backup}")
        for f in os.listdir(OUT):
            if f.endswith(".scp"):
                os.remove(os.path.join(OUT, f))
    made = build()
    for m in made:
        print("  wrote", m)
    print(f"\nwrote {len(made)} presets to {OUT}")
    if problems:
        print(f"\n{len(problems)} skipped / flagged:")
        for p in problems:
            print("  -", p)
