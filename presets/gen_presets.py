#!/usr/bin/env python3
"""Regenerate the StreamDiffusion .scp presets.

Port ids come from LibreDiffusion.hpp's inputs_t order under Crousti/ProcessModelPortInit.hpp's
`int inlet = 0` counter (verified by the portids probe): the two texture inlets and the two plain
value inlets (Embedding, Trigger) consume 0..3, so the first saved control (Workflow) is 4.
The 2026-06-07 presets predate the Embedding and LoRA-scale ports and are off by one/two throughout.

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
         klein_quality=22, rife_exp=23, async_=24, pacing=25)

# Workflow enum (LibreDiffusion.hpp)
(SD_T2I, SD_I2I, SD_T2I_CN, SD_I2I_CN, SD_T2I_IP, SD_I2I_IP, TURBO_T2I, TURBO_I2I,
 SDXL_T2I, SDXL_I2I, SDXL_T2I_CN, SDXL_I2I_CN, V2V_T2I, V2V_I2I,
 KLEIN_T2I, KLEIN_I2I, KLEIN_INPAINT, I2I_TURBO) = range(18)

CFG_NONE, CFG_SELF, CFG_FULL, CFG_INIT = 0, 1, 2, 3

# Validated start-index schedules. txt2img starts from pure noise (idx 0); img2img starts mid-schedule
# so the encoded frame survives -- idx ~32 is where 1-2 LCM steps stay sharp AND content-preserving
# (SETTINGS_GUIDELINES.md). Multi-step lists match the golden matrix / captured clip sidecars.
T2I_SCHED = {1: "0", 2: "0, 25", 3: "0, 16, 33", 4: "0, 12, 25, 37"}
I2I_SCHED = {1: "32", 2: "32, 45", 3: "20, 30, 40", 4: "16, 24, 32, 40"}

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


def preset(name, engines, workflow, *, steps=1, res=(512, 512), mode="t2i", timesteps=None,
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
    honours_cfg = workflow in (SD_T2I, SD_I2I, SD_T2I_CN, SD_I2I_CN, SD_T2I_IP, SD_I2I_IP)
    if timesteps is None:
        timesteps = (I2I_SCHED if is_i2i else T2I_SCHED)[steps]
    if cfg is None:
        cfg = CFG_SELF if (is_i2i and honours_cfg) else CFG_NONE
    if guidance is None:
        guidance = 1.2 if (is_i2i and honours_cfg) else 1.0
    if not honours_cfg:
        cfg, guidance = CFG_NONE, 1.0
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
        ("SD15 SDXS 512",            f"{E}/44549d33ab82fefb", 1, (512, 512)),
        ("SD15 Hyper-SD 1step",      f"{E}/32fde15b1bc3abf8", 1, (512, 512)),
        ("SD15 Hyper-SD 2step",      f"{E}/a93990bd296e3fe9", 2, (512, 512)),
        ("SD15 Hyper-SD 4step",      f"{E}/c049cd0e1923be58", 4, (512, 512)),
        ("SD15 LCM Dreamshaper 2step", f"{E}/611a46b8d791e296", 2, (512, 512)),
        ("SD15 LCM Dreamshaper 4step", f"{E}/5df487d86a94d462", 4, (512, 512)),
        ("SD21 base 1step",  "/media/data1/lrd-benchmark/engines/sd21-base-1step", 1, (512, 512)),
        ("SD21 base 4step",  "/media/data1/lrd-benchmark/engines/sd21-base-4step", 4, (512, 512)),
    ]
    for name, eng, steps, res in sd15:
        db = steps > 1          # 1.45x at 2 steps / 1.6x at 4, and it costs no extra VRAM
        made += [preset(f"{name} txt2img", eng, SD_T2I, steps=steps, res=res, mode="t2i",
                        denoise_batch=db),
                 preset(f"{name} img2img", eng, SD_I2I, steps=steps, res=res, mode="i2i",
                        denoise_batch=db)]

    # ---- SD-Turbo: 1024-dim CLIP, pad 0, single-step by construction (the node forces both).
    for name, eng, res in [
        ("SD-Turbo 512",      f"{E}/abb77149636ece27", (512, 512)),
        ("SD-Turbo 512x768",  f"{E}/e8f27dfc7f88d76f", (512, 768)),
        ("SD-Turbo 768x512",  f"{E}/1550845aa0f77668", (768, 512)),
        ("SD-Turbo 768",      f"{E}/fcf0c5cde3f4ee27", (768, 768)),
    ]:
        made += [preset(f"{name} txt2img", eng, TURBO_T2I, steps=1, res=res, mode="t2i"),
                 preset(f"{name} img2img", eng, TURBO_I2I, steps=1, res=res, mode="i2i")]

    # ---- SDXL family: 2048-dim CLIP + clip2, pooled embeds. The node pins cfg-none/guidance 0 on
    # this path, so the img2img presets cannot use the cfg-self restyle recipe.
    sdxl = [
        ("SDXL Turbo 512",           f"{E}/b34b954f2133d4f0", 1, (512, 512)),
        ("SDXL Turbo 1024",          f"{E}/b7ee298aa818aa51", 1, (1024, 1024)),
        ("SDXL Hyper-SD 4step",      f"{E}/847eb1390454cf93", 4, (1024, 1024)),
        ("SDXL Lightning 4step",     f"{E}/c8d002f4bf80148a", 4, (1024, 1024)),
        ("SDXL LCM-LoRA 2step",      f"{E}/d652fc8897e4587c", 2, (1024, 1024)),
        ("SDXL LCM-LoRA 2step 512",  f"{E}/f5a48e2ed23d671c", 2, (512, 512)),
        ("SDXL LCM-LoRA 3step",      f"{E}/0f1da52837e59afa", 3, (1024, 1024)),
        ("SDXL LCM-LoRA 4step",      f"{E}/40ddf8ec5cec3718", 4, (1024, 1024)),
        ("SDXL Segmind VegaRT 2step", "/media/data1/lrd-engines/vega", 2, (1024, 1024)),
    ]
    for name, eng, steps, res in sdxl:
        slow = res[0] >= 1024      # ~9-17 fps at 1024: diffuse off the render thread
        db = steps > 1             # the form the multi-step SDXL goldens were produced with
        made += [preset(f"{name} txt2img", eng, SDXL_T2I, steps=steps, res=res, mode="t2i",
                        cfg=CFG_NONE, guidance=1.0, denoise_batch=db, is_async=slow),
                 preset(f"{name} img2img", eng, SDXL_I2I, steps=steps, res=res, mode="i2i",
                        cfg=CFG_NONE, guidance=1.0, denoise_batch=db, is_async=slow)]

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
        made += [preset(f"{name} reinvent", eng, SD_T2I_CN, steps=1, res=(512, 512),
                        mode="t2i", prompt_key="cn_reinvent", check=False),
                 preset(f"{name} restyle", eng, SD_I2I_CN, steps=1, res=(512, 512),
                        mode="i2i", prompt_key="cn_restyle", check=False)]
    for name, eng in [("SDXL ControlNet Canny", "/media/data2/cn-sdxl-test/engines"),
                      ("SDXL ControlNet OpenPose", "/media/data2/cn-sdxl-openpose/engines")]:
        made += [preset(f"{name} reinvent", eng, SDXL_T2I_CN, steps=1, res=(1024, 1024),
                        mode="t2i", cfg=CFG_NONE, guidance=1.0, prompt_key="cn_reinvent",
                        is_async=False, check=False),
                 preset(f"{name} restyle", eng, SDXL_I2I_CN, steps=1, res=(1024, 1024),
                        mode="i2i", cfg=CFG_NONE, guidance=1.0, prompt_key="cn_restyle",
                        is_async=False, check=False)]

    # ---- IP-Adapter: the style image goes on the "Control / Style" inlet and is encoded on-device.
    made += [preset("SD15 IP-Adapter txt2img", "/media/data2/ip-sd15-test/engines", SD_T2I_IP,
                    steps=1, res=(512, 512), mode="t2i", prompt_key="ip", check=False),
             preset("SD15 IP-Adapter img2img", "/media/data2/ip-sd15-test/engines", SD_I2I_IP,
                    steps=1, res=(512, 512), mode="i2i", prompt_key="ip", check=False)]

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
             preset("FLUX2 Klein landscape", K, KLEIN_I2I, steps=1, res=(576, 320), mode="i2i",
                    timesteps="15, 25", prompt_key="klein", klein_quality=1, rife_exp=2,
                    is_async=True, check=False)]

    # ---- img2img-turbo (skip-VAE pix2pix-turbo): geometry and step count come from the engine.
    for name, eng in [("img2img-turbo edge2image", "/media/data1/img2img-turbo/bundle-traintest"),
                      ("img2img-turbo sketch2image", "/media/data1/img2img-turbo/bundle-sketch")]:
        made += [preset(name, eng, I2I_TURBO, steps=1, res=(512, 512), mode="i2i",
                        timesteps="0", prompt_key="i2it", check=False)]

    # ---- runtime-LoRA demo: engine exported with --lora PATH:runtime, so LoRA scale is live.
    made += [preset("SDXL crayon runtime-LoRA txt2img", "/media/data1/lora-test/crayon-rt",
                    SDXL_T2I, steps=1, res=(1024, 1024), mode="t2i", cfg=CFG_NONE, guidance=1.0,
                    lora_scale=1.0, is_async=True)]

    return [m for m in made if m]


if __name__ == "__main__":
    if not os.path.isdir(OUT):
        sys.exit(f"preset dir not found: {OUT}")
    backup = OUT + ".bak-2026-07-26"
    if not os.path.exists(backup):
        shutil.copytree(OUT, backup)
        print(f"backed up existing presets -> {backup}")
    for f in os.listdir(OUT):
        if f.endswith(".scp"):
            os.remove(os.path.join(OUT, f))
    made = build()
    print(f"\nwrote {len(made)} presets")
    if problems:
        print(f"\n{len(problems)} skipped / flagged:")
        for p in problems:
            print("  -", p)
