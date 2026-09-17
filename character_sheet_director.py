"""Muse Character Sheet Director.

Single-node wrapper around the "Krea2 Character Sheet - Aligned Views" workflow:
takes a 5-panel mannequin pose guide + one character reference photo, generates
five Krea2-edit views (one per pose) with the same models the original workflow
used, and lets the user confirm or re-roll (new seed) each view individually from
the node's own UI before assembling the final 4096x2304 sixteen-by-nine sheet.

Architecture note (this is a hard ComfyUI platform constraint, not a choice):
Python can only act from inside a real node execution triggered by a /prompt
submission. There is no live server->browser->server pause/resume inside one
execute() call. So every user action (initial run, re-roll one pose, finalize)
is its own small /prompt submission from the JS side; state that must survive
between those submissions (which poses are generated, their seeds, confirmed
flags) rides in the hidden `state_json` widget (serialized with the node) plus
an in-process cache here keyed by the node's unique_id, mirroring the pattern
used by Muse-MiniMax-Seed-Hunt-Studio / Director in this same custom_nodes tree.
"""
import hashlib
import json
import random

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes as comfy_nodes
from comfy_execution.graph import ExecutionBlocker

POSE_NAMES = ["01_portrait", "02_front", "03_left_profile", "04_right_profile", "05_back"]
POSE_LABELS = ["Portrait (close-up)", "Front", "Left profile", "Right profile", "Back"]
DEFAULT_SEEDS = [41001, 41002, 41003, 41004, 41005]

# Pixel crop rects for each pose panel, tuned to the original 1670x942 mannequin
# guide sheet. Scaled proportionally at runtime to whatever size guide_image is.
REFERENCE_GUIDE_SIZE = (1670, 942)  # (width, height)
GUIDE_CROPS = {
    "01_portrait":      dict(x=0,    y=0, width=508, height=942),
    "02_front":         dict(x=508,  y=0, width=307, height=942),
    "04_right_profile": dict(x=815,  y=0, width=237, height=942),
    "03_left_profile":  dict(x=1052, y=0, width=267, height=942),
    "05_back":          dict(x=1319, y=0, width=351, height=942),
}

# Target (latent-space) pixel resolution per pose. Portrait keeps its own native
# size. Front/back get the full-width canvas since a front-on silhouette is
# genuinely that wide (shoulder to shoulder). [2026-09-22] Left/right profile
# used to share that same 544-wide canvas, but a true side-on silhouette is
# only chest-to-back deep - much narrower than shoulder width - so the model
# was widening/stockifying the body to fill a frame built for a front-on
# pose (Andy: "they don't look like they represent the normal height of the
# person"). Narrowed to 416 (divisible by 16 for Klein's EmptyFlux2LatentImage
# requirement) with enough margin left for an outstretched arm/bag strap/hair
# without clipping. Height is unchanged, so MuseSheetAlignFigure's fixed
# figure_height/bottom_margin in ALIGN_BODY_KW still lines every panel up at
# the same final height - only the pre-alignment generation canvas is
# narrower for these two poses. Reference-image conditioning (guide_for_gen,
# character_image) is resolution-independent of this (Krea2EditNormalizedAttentionGuidance's
# fit_mode="fit" re-derives its reference natively from source_image+vae
# rather than requiring source_latent to match target_latent's shape -
# confirmed by reading krea2-nag/nodes.py), so no other alignment/crop
# constant needs to change alongside this.
TARGET_SIZE = {
    "01_portrait": (544, 976),
    "02_front": (544, 1784),
    "03_left_profile": (416, 1784),
    "04_right_profile": (416, 1784),
    "05_back": (544, 1784),
}

ALIGN_GUIDE_KW = dict(width=544, height=1784, figure_height=1584, bottom_margin=100, threshold=0.1)
ALIGN_BODY_KW = dict(width=704, height=2304, figure_height=2048, bottom_margin=128, threshold=0.1)
PORTRAIT_FINAL_SIZE = (1280, 2304)  # (width, height)

# [2026-09-17] Named presets showing the actual final pixel size, not a bare
# scale multiplier - ported from the Klein sibling node's identical feature
# ("who's supposed to know that's the default size"). Every panel's
# proportions stay identical across presets (see _assemble_final) - the
# character sheet is still a fixed, exact size, you're just picking which
# fixed size. Standard (the original size) is deliberately the largest, with
# three smaller options below it.
OUTPUT_SIZE_PRESETS = {
    "4096x2304 (Standard - default)": 1.0,
    "3072x1728 (Medium)": 0.75,
    "2048x1152 (Small)": 0.5,
    "1536x864 (Extra Small)": 0.375,
}

# [2026-09-16] Replaced with Andy's SatoDive-workflow negative (merged with our
# own, multi-panel-only terms like "missing back view"/"turnaround" dropped
# since each call here only ever renders one pose, not a composite sheet).
# Notably adds "hand on the face / fingers close to face" - the exact defect
# seen bleeding through from the character reference on the portrait pose.
NEGATIVE_TEXT = (
    "gray mannequin skin, gray plastic, sculpture, plastic mannequin, extra people, "
    "duplicate figure, merged figures, extra limbs, cropped head, cropped feet, cut off frame, "
    "touching image border, edge clipping, off-center, clipped clothing, missing garment layers, "
    "altered outfit, mismatched clothes, text, watermark, signature, blurry, low quality, "
    "blurry edges, soft edges, color bleeding, feathering, shadows on background, dirty "
    "background, dark backdrop, hand on the face, fingers on the head, fingers close to face, "
    "hands covering face, hands touching, hands on chest, bent elbows, crossed arms, folded "
    "arms, arms crossed over chest"
)

# [2026-09-17] This pipeline runs KSampler at cfg=1, where negative conditioning
# is mathematically a no-op (cfg=1 collapses the CFG formula to just the
# positive branch) - so KSampler's own "negative" (Krea2EditGroundedEncode with
# an empty prompt) never did any steering. The only real negative-steering path
# is NAG's nag_negative. It used to be one shared block of text for all 5
# poses, which can't express "no collar/zip" for the back view without also
# suppressing the collar/zip on views that need it (Portrait, Front). Per-pose
# extra negative terms, appended only for the poses listed here and encoded
# separately (see _get_neg_cond), fix that without touching the shared base.
POSE_EXTRA_NEGATIVE = {
    "05_back": (
        "front zip, front buttons, front collar, low neckline, cleavage, visible chest, "
        "front closure, front pockets, front of garment"
    ),
}

# [2026-09-17] ref_boost pulls attention toward literally reproducing the
# character reference's visual content, not just its identity/color - at the
# shared default (4.0), that's strong enough to make the Back pose reproduce
# the reference's front zip/collar shape verbatim regardless of what the
# prompt or negative say (confirmed: prompt rewrites and per-pose negative
# terms alone did not stop it). Per-pose multipliers applied to the user's
# ref_boost widget value let Back anchor less literally to the reference's
# exact garment structure while the other 4 poses (working fine) are
# unaffected.
POSE_REF_BOOST_MULTIPLIER = {
    "05_back": 0.5,
}

# [2026-09-17] Replaced again with Andy's own simpler prompt style, which he
# confirmed working (correct white background, outfit and identity) for
# Portrait/Right profile/Back in real testing - the SatoDive-derived template
# above is kept only in comments/memory for reference, not used any more. Same
# structure on every pose ("Use image 1 for... replace with the person on
# image 2... Retain ALL clothing/hair... on white background with..."), only
# the final framing clause changes per pose. Back is the one exception: it
# drops "Retain ALL clothing" (which was pushing the model to literally copy
# the reference's FRONT-visible clothing details - zip/collar/print - onto a
# view where those details shouldn't be visible) and explicitly says not to
# repeat front-specific garment details; that's paired with a per-pose NAG
# negative for the same reason (see POSE_EXTRA_NEGATIVE / _get_neg_cond) since
# cfg=1 makes KSampler's own negative a no-op - text alone couldn't fix this.
POSE_PROMPTS = {
    "01_portrait": (
        "Use image 1 for the body orientation and crop. Then replace image with the exact "
        "person on image 2. Retain ALL clothing, facial features and hair. Final image should "
        "be on white backgrounds with arms by the side, zoomed in to head and shoulders with "
        "shoulders straight ahead."
    ),
    "02_front": (
        "Use image 1 for the body orientation and crop. Then replace image with the exact "
        "person on image 2. Retain ALL clothing, facial features and hair. Final image should "
        "be on white backgrounds with arms by the side, full body visible from head to toe, "
        "standing straight facing forward."
    ),
    "03_left_profile": (
        "Use image 1 for the body orientation and crop. Then replace image with the exact "
        "person on image 2. Retain ALL clothing, facial features and hair. Final image should "
        "be on white backgrounds with arms by the side, full body visible from head to toe, "
        "body turned to show a side profile facing left."
    ),
    "04_right_profile": (
        "Use image 1 for the body orientation and crop. Then replace image with the exact "
        "person on image 2. Retain ALL clothing, facial features and hair. Final image should "
        "be on white backgrounds with arms by the side, full body visible from head to toe, "
        "body turned to show a side profile facing right."
    ),
    "05_back": (
        "Use image 1 for the body orientation and crop. Then replace image with the rear view "
        "of the person on image 2. Keep the same hair, outfit colour and fabric, but show a "
        "plausible back of the garment - if the front has a zip, buttons, collar or print, do "
        "not repeat them on the back. Final image should be on white backgrounds with arms by "
        "the side, full body visible from head to toe, back of the body facing the camera."
    ),
}

RMBG_KW = dict(
    model="RMBG-2.0", sensitivity=1.0, process_res=1024, mask_blur=0, mask_offset=0,
    invert_output=False, refine_foreground=False, background="Color", background_color="#ffffff",
)

DEFAULT_PROMPTS = [POSE_PROMPTS[name] for name in POSE_NAMES]
DEFAULT_STATE = {
    "seeds": list(DEFAULT_SEEDS), "confirmed": [False] * 5, "action": None,
    "prompts": list(DEFAULT_PROMPTS),
}

# Face-detail (Impact Pack FaceDetailer) pass, run on each pose right after decode.
# detector_type -> the ultralytics model file that already ships in this install.
FACE_DETAIL_DETECTOR_MODELS = {
    "face": "bbox/face_yolov8m.pt",
    "hand": "bbox/hand_yolov8s.pt",
    "person": "segm/person_yolov8m-seg.pt",
}
FACE_DETAIL_STEPS = 4  # matches Andy's proven standalone FaceDetailer setup
# Everything below is locked to Andy's own proven FaceDetailer settings (screenshot
# 2026-09-15) - only detector type / sampler / scheduler / denoise are exposed as
# node controls; the rest would just be clutter for a setup that already works.
# [2026-09-17] Rebuilt on Impact Pack's SEGS pipeline instead of calling
# FaceDetailer directly - see character_sheet_klein.py's identical rebuild
# for the full story. FaceDetailer has no built-in cap on how many detected
# regions it refines; the Klein sibling node hit a live 300-then-again-later
# false-positive detection storm on this exact same mechanism, each one
# running its own full encode/sample/decode pass. Ported the same fix here
# pre-emptively since this node has the identical vulnerability even though
# it hadn't malfunctioned yet (guide_size_for False = "crop_region").
FACE_DETAIL_BBOX_KW = dict(threshold=0.50, dilation=10, crop_factor=2.0, drop_size=10, labels="all")
FACE_DETAIL_DETAILER_KW = dict(guide_size=1536.0, guide_size_for=False, max_size=1536.0, noise_mask_feather=10)
FACE_DETAIL_PASTE_KW = dict(feather=10, alpha=255)

# Loaded-model cache, shared across node instances/sessions (reload is expensive;
# these rarely change between runs). Per-session generation state lives separately
# in _SESSIONS, keyed by the node's own unique_id.
_MODEL_CACHE = {"key": None}
_DETECTOR_CACHE = {"key": None}
_SESSIONS = {}


def _node(name):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"[MuseCharacterSheetDirector] Required node '{name}' is not registered. "
            f"Check that its custom_nodes package is installed and loaded."
        )
    return cls()


def _tensor_hash(t):
    arr = t.detach().to("cpu", torch.float32).contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _signature(guide_image, character_image, unet_name, identity_lora, filter_bypass_lora,
               clip_name, vae_name, ref_boost, ref_boost_a, steps, cfg,
               face_detail, face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise,
               model_override=None, clip_override=None, vae_override=None):
    payload = [
        _tensor_hash(guide_image), _tensor_hash(character_image),
        unet_name, identity_lora, filter_bypass_lora, clip_name, vae_name,
        round(float(ref_boost), 4), round(float(ref_boost_a), 4), int(steps), round(float(cfg), 4),
        bool(face_detail), face_detail_type, face_detail_sampler, face_detail_scheduler,
        round(float(face_detail_denoise), 4),
        # [2026-09-19] Deliberately NOT including id(model_override)/id(clip_override)/
        # id(vae_override) here anymore. The idea was that id() changing would mean
        # the upstream loader re-executed with a genuinely different model - but
        # object identity across separate /prompt submissions isn't reliable to key
        # a session-wipe off: a single reroll click on ONE pose (nothing about the
        # model config touched at all) still ended up invalidating the whole
        # session and forcing all 5 poses to regenerate (confirmed via ComfyUI's
        # own history: a 1-pose reroll took ~93s, essentially the same as a fresh
        # 5-pose run). A wrongly-missed model swap is a low-cost mistake (rerolling
        # a pose or two makes it obvious); a wrongly-forced full session wipe on
        # every single click defeats the entire point of the confirm/reroll UI.
        model_override is not None, clip_override is not None, vae_override is not None,
    ]
    return hashlib.sha1(json.dumps(payload).encode("utf-8")).hexdigest()


def _get_models(unet_name, identity_lora, filter_bypass_lora, clip_name, vae_name,
                 model_override=None, clip_override=None, vae_override=None):
    key = (
        unet_name, identity_lora, filter_bypass_lora, clip_name, vae_name,
        id(model_override) if model_override is not None else None,
        id(clip_override) if clip_override is not None else None,
        id(vae_override) if vae_override is not None else None,
    )
    if _MODEL_CACHE.get("key") == key:
        return _MODEL_CACHE

    print(f"[MuseCharacterSheetDirector] loading models: {key}", flush=True)
    if model_override is not None:
        # [2026-09-19] An incoming model (e.g. from Muse Model Loader) is
        # assumed to already have its own LoRAs baked in via that loader's own
        # LoRA slots - applying identity_lora/filter_bypass_lora again on top
        # would double them up, so the manual LoRA widgets are skipped
        # entirely whenever a model socket is connected.
        model = model_override
    else:
        model = _node("UNETLoader").load_unet(unet_name, "default")[0]
        model = _node("LoraLoaderModelOnly").load_lora_model_only(model, identity_lora, 1.0)[0]
        model = _node("LoraLoaderModelOnly").load_lora_model_only(model, filter_bypass_lora, 1.0)[0]
    clip = clip_override if clip_override is not None else _node("CLIPLoader").load_clip(clip_name, "krea2", "default")[0]
    vae = vae_override if vae_override is not None else _node("VAELoader").load_vae(vae_name)[0]
    neg_cond = _node("CLIPTextEncode").encode(clip, NEGATIVE_TEXT)[0]

    _MODEL_CACHE.clear()
    _MODEL_CACHE.update({
        "key": key, "model": model, "clip": clip, "vae": vae, "neg_cond": neg_cond,
        "extra_neg_cond": {},
    })
    return _MODEL_CACHE


def _get_neg_cond(models, pose_name):
    extra = POSE_EXTRA_NEGATIVE.get(pose_name)
    if not extra:
        return models["neg_cond"]
    cache = models["extra_neg_cond"]
    if pose_name not in cache:
        cache[pose_name] = _node("CLIPTextEncode").encode(models["clip"], NEGATIVE_TEXT + ", " + extra)[0]
    return cache[pose_name]


def _get_detector(detector_type):
    model_name = FACE_DETAIL_DETECTOR_MODELS[detector_type]
    if _DETECTOR_CACHE.get("key") == model_name:
        return _DETECTOR_CACHE
    print(f"[MuseCharacterSheetDirector] loading detector: {model_name}", flush=True)
    bbox_detector, _segm_detector = _node("UltralyticsDetectorProvider").doit(model_name)
    _DETECTOR_CACHE.clear()
    _DETECTOR_CACHE.update({"key": model_name, "bbox_detector": bbox_detector})
    return _DETECTOR_CACHE


def _scaled_crop(pose_name, guide_image):
    ref_w, ref_h = REFERENCE_GUIDE_SIZE
    actual_h, actual_w = guide_image.shape[1], guide_image.shape[2]
    sx, sy = actual_w / ref_w, actual_h / ref_h
    rect = GUIDE_CROPS[pose_name]
    x = int(round(rect["x"] * sx))
    y = int(round(rect["y"] * sy))
    w = int(round(rect["width"] * sx))
    h = int(round(rect["height"] * sy))
    x = max(0, min(x, actual_w - 1))
    y = max(0, min(y, actual_h - 1))
    w = max(1, min(w, actual_w - x))
    h = max(1, min(h, actual_h - y))
    return x, y, w, h


def _crop(image, x, y, w, h):
    return image[:, y:y + h, x:x + w, :]


def _ensure_rgb(image):
    # Krea2EditGroundedEncode / MuseSheetAlignFigure / RMBG all expect a plain
    # 3-channel IMAGE; LoadImage can hand back RGBA depending on the source file.
    return image[..., :3] if image.shape[-1] > 3 else image


def _resize(image, width, height, method="lanczos"):
    samples = image.movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return samples.movedim(1, -1)


def _face_detail(image, nag_model, positive, negative, models, seed, cfg,
                  face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise):
    """[2026-09-17] SEGS pipeline, not a direct FaceDetailer call - see the
    FACE_DETAIL_* constants' comment for why. detect (BboxDetectorSEGS) ->
    keep only the single largest region (ImpactSEGSOrderedFilter,
    take_count=1) -> refine just that one (SEGSDetailer) -> composite back
    onto the full image (SEGSPaste). However many "faces" the detector
    reports - 1 or 300 - exactly one refine pass ever runs."""
    detector = _get_detector(face_detail_type)
    print(f"[MuseCharacterSheetDirector] >>> face-detail pass START (detect={face_detail_type}, "
          f"sampler={face_detail_sampler}, scheduler={face_detail_scheduler}, "
          f"denoise={face_detail_denoise}, steps={FACE_DETAIL_STEPS})", flush=True)
    raw_segs = _node("BboxDetectorSEGS").doit(
        bbox_detector=detector["bbox_detector"], image=image, **FACE_DETAIL_BBOX_KW,
    )[0]
    largest_seg, _ = _node("ImpactSEGSOrderedFilter").doit(
        segs=raw_segs, target="area(=w*h)", order=True, take_start=0, take_count=1,
    )
    found = len(largest_seg[1]) > 0
    if not found:
        print("[MuseCharacterSheetDirector] <<< face-detail pass END - NO region detected, image unchanged", flush=True)
        return image
    basic_pipe = _node("ToBasicPipe").doit(
        model=nag_model, clip=models["clip"], vae=models["vae"], positive=positive, negative=negative,
    )[0]
    refined_segs, _ = _node("SEGSDetailer").doit(
        image=image, segs=largest_seg, seed=int(seed), steps=FACE_DETAIL_STEPS, cfg=float(cfg),
        sampler_name=face_detail_sampler, scheduler=face_detail_scheduler, denoise=float(face_detail_denoise),
        noise_mask=True, force_inpaint=True, basic_pipe=basic_pipe, refiner_ratio=0.2, batch_size=1, cycle=1,
        **FACE_DETAIL_DETAILER_KW,
    )
    result_image = _node("SEGSPaste").doit(image=image, segs=refined_segs, **FACE_DETAIL_PASTE_KW)[0]
    print("[MuseCharacterSheetDirector] <<< face-detail pass END - region found and refined", flush=True)
    return result_image


def _generate_pose(pose_idx, guide_image, character_image, char_latent, models, seed, prompt,
                    ref_boost, ref_boost_a, steps, cfg,
                    face_detail, face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise):
    pose_name = POSE_NAMES[pose_idx]
    x, y, w, h = _scaled_crop(pose_name, guide_image)
    guide_crop = _crop(guide_image, x, y, w, h)

    target_w, target_h = TARGET_SIZE[pose_name]
    if pose_idx == 0:
        guide_for_gen = guide_crop
    else:
        guide_for_gen = _node("MuseSheetAlignFigure").align(image=guide_crop, mask=None, **ALIGN_GUIDE_KW)[0]

    target_latent = _node("EmptyLatentImage").generate(width=target_w, height=target_h, batch_size=1)[0]
    guide_latent = _node("VAEEncode").encode(vae=models["vae"], pixels=guide_for_gen)[0]

    pose_ref_boost = float(ref_boost) * POSE_REF_BOOST_MULTIPLIER.get(pose_name, 1.0)

    # Combined edit-patch + NAG in one call. The installed krea2-nag package's plain
    # Krea2NormalizedAttentionGuidance now REFUSES to stack after Krea2EditModelPatch
    # (raises ValueError telling you to use this combined node instead) - the source
    # workflow JSON's separate two-node wiring predates that guard and would no
    # longer run against the currently installed package. This combined node does
    # the same thing (source-reference attention on the positive path, NAG on the
    # target tokens only) in a single call.
    nag_model = _node("Krea2EditNormalizedAttentionGuidance").patch(
        model=models["model"], nag_negative=_get_neg_cond(models, pose_name), source_latent=guide_latent,
        phi=4.0, tau=2.5, alpha=0.25, sigma_start=1000.0, sigma_end=0.0,
        source_latent_b=char_latent, ref_boost=pose_ref_boost, ref_boost_a=float(ref_boost_a),
        fit_mode="fit", ref_boost_mask=None, vae=models["vae"],
        source_image=guide_for_gen, source_image_b=character_image, target_latent=target_latent,
    )[0]
    positive = _node("Krea2EditGroundedEncode").encode(
        clip=models["clip"], prompt=prompt,
        image=guide_for_gen, image_b=character_image, grounding_px=768, system_prompt="",
    )[0]
    negative = _node("Krea2EditGroundedEncode").encode(
        clip=models["clip"], prompt="",
        image=guide_for_gen, image_b=character_image, grounding_px=768, system_prompt="",
    )[0]
    sampled = _node("KSampler").sample(
        model=nag_model, seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name="euler", scheduler="simple",
        positive=positive, negative=negative, latent_image=target_latent, denoise=1.0,
    )[0]
    decoded = _node("VAEDecode").decode(vae=models["vae"], samples=sampled)[0]

    if face_detail:
        decoded = _face_detail(
            decoded, nag_model, positive, negative, models, seed, cfg,
            face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise,
        )

    # Background cleanup happens here, not just at final assembly, so the
    # per-pose preview the user reviews/confirms already shows the same clean
    # white background the finished sheet will have - otherwise confirming a
    # pose is a guess, since RMBG previously only ran once at the very end.
    clean_image, mask, _ = _node("RMBG").process_image(image=decoded, **RMBG_KW)
    return clean_image, mask


def _load_preview_pixels(state, index):
    """[2026-09-20] Core file lookup, split out of _restore_confirmed_preview
    so it can also be used as a BEST-EFFORT restore for the "edit" action
    (see run()) - that path has nothing wrong with it if there simply isn't a
    prior generation to edit yet (e.g. right after a restart wiped the
    in-process session), it just skips gracefully. Returns pixels or None,
    never raises - _restore_confirmed_preview below is what turns a miss into
    a hard failure, since a confirmed lock actually has to hold.
    """
    import os
    import numpy as np
    from PIL import Image
    previews = state.get("previews") or []
    item = previews[index] if index < len(previews) else None
    if not item or item.get("type") not in ("temp", "output"):
        return None
    base = folder_paths.get_temp_directory() if item["type"] == "temp" else folder_paths.get_output_directory()
    base = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base, item.get("subfolder", ""), item.get("filename", "")))
    if os.path.commonpath([base, path]) != base or not os.path.isfile(path):
        return None
    with Image.open(path) as im:
        return torch.from_numpy(np.array(im.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)


def _restore_confirmed_preview(state, index, seed, prompt):
    """Reload a confirmed pose's accepted pixels from its saved preview file
    instead of silently regenerating it. A confirmed pose is a lock - that
    lock has to survive both a settings change (which invalidates the
    in-process session cache, see run()) and a ComfyUI restart (which wipes
    the in-process cache entirely, since _SESSIONS is just a module dict).
    [2026-09-17/18] Diagnosed and initially fixed by Codex after a credit
    limit cut off its own session; finished and merged in here, with one
    correctness fix: the original version left `mask` as None on a restored
    pose, which silently degraded the body-alignment step (4 of 5 poses need
    a real foreground mask, not the crude fallback MuseSheetAlignFigure uses
    when none is given) for any pose that survived a restart. Re-deriving the
    mask via RMBG here (cheap, and only runs once per pose per restore) fixes
    that gap.
    """
    pixels = _load_preview_pixels(state, index)
    if pixels is None:
        raise ValueError(
            f"Confirmed pose {index + 1}'s saved preview file is missing on disk "
            f"(temp previews can get cleaned up over time). Unconfirm it to regenerate."
        )
    # The saved image already had RMBG applied (white background) before it was
    # written - re-run RMBG here just to recover a matching foreground mask,
    # not to redo the background cleanup itself.
    _, mask, _ = _node("RMBG").process_image(image=pixels, **RMBG_KW)
    return {"seed": seed, "prompt": prompt, "image": pixels, "mask": mask}


def _edit_pose(pose_idx, source_image, character_image, instruction, char_latent, models, seed,
               ref_boost, ref_boost_a, steps, cfg,
               face_detail, face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise):
    """[2026-09-20] Targeted follow-up edit on a pose's OWN already-generated
    pixels (e.g. "add high heel shoes") - not a fresh regenerate from the
    guide/character references. Krea2Edit is an edit model just like Klein,
    so this is the same mechanism as _generate_pose: source_image is this
    pose's own prior output (standing in for the cropped/aligned guide crop),
    instruction stands in for the pose's base prompt. character_image stays
    wired in as the second reference (identity/outfit anchor), same as normal
    generation - an edit like "add shoes" still needs to know whose shoes.
    The pose's own stored base prompt is left untouched by this (see the
    "edit" action handling in run()), so a later normal re-roll still
    regenerates from the original pose description, not from whatever
    one-off edit instruction was typed here.
    """
    pose_name = POSE_NAMES[pose_idx]
    target_w, target_h = TARGET_SIZE[pose_name]
    target_latent = _node("EmptyLatentImage").generate(width=target_w, height=target_h, batch_size=1)[0]
    source_latent = _node("VAEEncode").encode(vae=models["vae"], pixels=source_image)[0]

    pose_ref_boost = float(ref_boost) * POSE_REF_BOOST_MULTIPLIER.get(pose_name, 1.0)

    nag_model = _node("Krea2EditNormalizedAttentionGuidance").patch(
        model=models["model"], nag_negative=_get_neg_cond(models, pose_name), source_latent=source_latent,
        phi=4.0, tau=2.5, alpha=0.25, sigma_start=1000.0, sigma_end=0.0,
        source_latent_b=char_latent, ref_boost=pose_ref_boost, ref_boost_a=float(ref_boost_a),
        fit_mode="fit", ref_boost_mask=None, vae=models["vae"],
        source_image=source_image, source_image_b=character_image, target_latent=target_latent,
    )[0]
    positive = _node("Krea2EditGroundedEncode").encode(
        clip=models["clip"], prompt=instruction,
        image=source_image, image_b=character_image, grounding_px=768, system_prompt="",
    )[0]
    negative = _node("Krea2EditGroundedEncode").encode(
        clip=models["clip"], prompt="",
        image=source_image, image_b=character_image, grounding_px=768, system_prompt="",
    )[0]
    sampled = _node("KSampler").sample(
        model=nag_model, seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name="euler", scheduler="simple",
        positive=positive, negative=negative, latent_image=target_latent, denoise=1.0,
    )[0]
    decoded = _node("VAEDecode").decode(vae=models["vae"], samples=sampled)[0]

    if face_detail:
        decoded = _face_detail(
            decoded, nag_model, positive, negative, models, seed, cfg,
            face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise,
        )

    clean_image, mask, _ = _node("RMBG").process_image(image=decoded, **RMBG_KW)
    return clean_image, mask


def _assemble_final(sess, models, output_scale=1.0):
    """[2026-09-17] output_scale (default 1.0 = the original 4096x2304)
    uniformly scales every panel's dimensions - portrait width/height and the
    body panels' width/height/figure_height/bottom_margin all move together,
    so proportions stay identical to the tuned defaults; only the overall
    size changes. Ported from the Klein sibling node's identical feature."""
    scale = float(output_scale)
    portrait_size = (round(PORTRAIT_FINAL_SIZE[0] * scale), round(PORTRAIT_FINAL_SIZE[1] * scale))
    align_kw = {**ALIGN_BODY_KW,
                "width": round(ALIGN_BODY_KW["width"] * scale),
                "height": round(ALIGN_BODY_KW["height"] * scale),
                "figure_height": round(ALIGN_BODY_KW["figure_height"] * scale),
                "bottom_margin": round(ALIGN_BODY_KW["bottom_margin"] * scale)}
    panels = []
    for i in range(5):
        pose = sess["poses"][i]
        if i == 0:
            panel = _resize(pose["image"], *portrait_size)
        else:
            panel = _node("MuseSheetAlignFigure").align(image=pose["image"], mask=pose["mask"], **align_kw)[0]
        panels.append(panel)
    return torch.cat(panels, dim=2)


def _preview(image, persist=False):
    # [2026-09-19] PreviewImage always writes to ComfyUI's temp dir, which is
    # ephemeral by design (wiped by a restart or periodic cleanup) - fine for
    # an unconfirmed pose's rolling preview, since losing it just means it
    # regenerates. A CONFIRMED pose is a lock, though, and the lock has to
    # keep working after a restart - SaveImage (same save_images() signature,
    # PreviewImage is literally a subclass of it) writes to the permanent
    # output dir instead, so _restore_confirmed_preview() always has a real
    # file to load from. See the pose_previews line in run() for what decides
    # persist per pose, and the confirm button's queue() in the JS for what
    # guarantees a run happens right when a pose gets locked, not whenever the
    # next unrelated run happens to occur.
    node = _node("SaveImage") if persist else _node("PreviewImage")
    kwargs = {"images": image}
    if persist:
        kwargs["filename_prefix"] = "MuseCharacterSheet"
    result = node.save_images(**kwargs)
    return result["ui"]["images"][0]


class MuseCharacterSheetDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_image": ("IMAGE", {"tooltip": "5-panel mannequin pose guide sheet (portrait, front, left, right, back)."}),
                "character_image": ("IMAGE", {"tooltip": "Reference photo of the character/outfit, used for every pose."}),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "identity_lora": (folder_paths.get_filename_list("loras"),),
                "filter_bypass_lora": (folder_paths.get_filename_list("loras"),),
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "ref_boost": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "ref_boost_a": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "face_detail": ("BOOLEAN", {"default": True, "tooltip": "Run a FaceDetailer pass on each pose to fix soft/plastic faces."}),
                "face_detail_type": (list(FACE_DETAIL_DETECTOR_MODELS.keys()), {"default": "face"}),
                "face_detail_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "face_detail_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "face_detail_denoise": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_mode": (["random", "fixed"], {"default": "random", "tooltip": "What starting seeds a fresh run gets AFTER a completed sheet resets (not the per-pose New seed button, which always picks a fresh random seed regardless). 'random' means running again with the same guide/character photos produces different poses without needing new images; 'fixed' always starts from the same seeds (41001-41005), so a re-run reproduces the same result."}),
                "state_json": ("STRING", {"multiline": True, "default": json.dumps(DEFAULT_STATE)}),
                # [2026-09-17] Appended AFTER state_json deliberately, never
                # inserted mid-list - widgets_values on an already-saved
                # workflow is positional, not by-name, so a mid-list insert
                # silently shifts every later widget's stored value onto the
                # wrong slot (this bit hard on the Klein sibling node - see
                # its own comment here). New widgets always go at the true
                # end from now on.
                "output_size": (list(OUTPUT_SIZE_PRESETS.keys()), {"default": "4096x2304 (Standard - default)", "tooltip": "Final assembled sheet size - pick the exact pixel dimensions you want. Every panel's proportions stay identical across presets, only the overall size changes. Only affects the final Build step, not the per-pose generation/preview resolution."}),
            },
            # [2026-09-19] Optional overrides, e.g. from Muse Model Loader - purely
            # additive, the manual unet_name/identity_lora/filter_bypass_lora/
            # clip_name/vae_name widgets above still work standalone when nothing
            # is plugged in here. When a socket IS connected it wins over the
            # matching widget(s); see _get_models(). LoRAs are assumed to already
            # be baked into an incoming model, so they travel with it automatically
            # rather than needing their own separate socket.
            "optional": {
                "model_override": ("MODEL", {"tooltip": "Optional. Overrides unet_name/identity_lora/filter_bypass_lora when connected - LoRAs baked into this model travel with it."}),
                "clip_override": ("CLIP", {"tooltip": "Optional. Overrides clip_name when connected."}),
                "vae_override": ("VAE", {"tooltip": "Optional. Overrides vae_name when connected."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("character_sheet",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Muse/Character Sheet"
    DESCRIPTION = (
        "Generates a 5-pose Krea2 character sheet from a mannequin guide + character photo. "
        "Confirm or re-roll each pose in the node's own UI; once all five are confirmed it "
        "assembles the final 4096x2304 sheet."
    )

    def run(self, guide_image, character_image, unet_name, identity_lora, filter_bypass_lora,
            clip_name, vae_name, ref_boost, ref_boost_a, steps, cfg,
            face_detail, face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise,
            seed_mode, state_json, output_size, unique_id, model_override=None, clip_override=None, vae_override=None):
        guide_image = _ensure_rgb(guide_image)
        character_image = _ensure_rgb(character_image)
        try:
            state = json.loads(state_json) if state_json else {}
        except (TypeError, ValueError):
            state = {}
        seeds = [int(s) for s in (state.get("seeds") or DEFAULT_SEEDS)]
        confirmed = list(state.get("confirmed") or [False] * 5)
        prompts = list(state.get("prompts") or DEFAULT_PROMPTS)
        if len(prompts) != 5:
            prompts = list(DEFAULT_PROMPTS)
        action = state.get("action")

        sig = _signature(guide_image, character_image, unet_name, identity_lora,
                          filter_bypass_lora, clip_name, vae_name, ref_boost, ref_boost_a, steps, cfg,
                          face_detail, face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise,
                          model_override, clip_override, vae_override)
        sess = _SESSIONS.setdefault(str(unique_id), {})
        if sess.get("sig") != sig:
            # [2026-09-18] Confirmed poses are a lock, not just a cache entry -
            # a settings change (ref_boost, steps, a new model, etc.) used to
            # wipe the whole session including anything already confirmed,
            # silently un-confirming poses the user had already accepted.
            # Retain confirmed poses' generated data across the reset instead;
            # everything else still regenerates as before, since it's the
            # settings themselves that changed for them.
            retained = {i: p for i, p in sess.get("poses", {}).items() if i < len(confirmed) and confirmed[i]}
            sess.clear()
            sess["sig"] = sig
            sess["poses"] = retained
            # seeds/confirmed are NOT reset here for the same reason - a
            # confirmed pose's seed/confirmed flag has to survive this reset
            # for the retained pose above to still make sense. Un-confirmed
            # poses simply regenerate with whatever seed they already had.
            # [2026-09-19] A pending finalize is NOT cleared here anymore -
            # confirmed data via the log above proves this: click Confirm on
            # all 5, click Build final sheet, sig happens to have drifted
            # (e.g. a different upstream LoadImage node cached in) between
            # generating and finalizing - the finalize action got silently
            # discarded right here, so run() fell through to a plain "ready"
            # status and nothing was built. A stale RE-ROLL genuinely doesn't
            # make sense after settings changed (it names a pose/seed pairing
            # that may no longer be relevant), so that's still cleared - but
            # finalize only ever reads whatever's already confirmed/retained
            # above, which is exactly the same data regardless of what
            # triggered this particular sig change.
            if not (action and action.get("type") == "finalize"):
                action = None
            # prompts are NOT reset here either - they're pose-instruction
            # templates, not tied to a specific character/guide image, so an
            # edited prompt should survive swapping to a different reference photo.

        # A confirmed pose whose generated data isn't in the session cache
        # (lost to a ComfyUI restart, which wipes _SESSIONS entirely, or to
        # the retain-on-reset above skipping it because it wasn't confirmed
        # yet at the time) gets its accepted pixels reloaded from disk rather
        # than silently regenerated - the lock must hold even across a restart.
        for i in range(5):
            if confirmed[i] and sess["poses"].get(i) is None:
                sess["poses"][i] = _restore_confirmed_preview(state, i, seeds[i], prompts[i])

        models = _get_models(unet_name, identity_lora, filter_bypass_lora, clip_name, vae_name,
                              model_override, clip_override, vae_override)

        if "char_latent" not in sess:
            sess["char_latent"] = _node("VAEEncode").encode(vae=models["vae"], pixels=character_image)[0]

        if action and action.get("type") == "reroll":
            i = int(action["pose"])
            if not confirmed[i]:  # a locked pose ignores a stray reroll request
                seeds[i] = int(action["seed"])

        def run_edit(i, source_image, instruction):
            """Runs _edit_pose and stores the result, tagged with the
            instruction and the SOURCE it was applied to (edit_source_image) -
            that tag is what makes reroll_edit() below possible: re-running
            the same instruction with a new seed from the same starting
            point, instead of stacking onto whatever the previous edit
            produced. A plain fresh generation (the normal to_generate loop)
            always overwrites sess["poses"][i] with a brand-new dict that has
            no edit_instruction/edit_source_image keys at all, so an old edit
            tag can never survive into a genuinely fresh pose by accident.
            [2026-09-23] Deliberately does NOT fold the pose's base prompt in
            here (an earlier attempt at that did) - every POSE_PROMPTS entry
            contains "Retain ALL clothing, facial features and hair", which
            directly contradicts any edit instruction that changes clothing,
            and was confirmed to cause visible color bleed/merging (Andy: "the
            weird saturated color... merging things together"). instruction
            alone is what actually gets sent as text conditioning."""
            image, mask = _edit_pose(
                i, source_image, character_image, instruction, sess["char_latent"], models,
                seeds[i], ref_boost, ref_boost_a, steps, cfg,
                face_detail, face_detail_type, face_detail_sampler, face_detail_scheduler, face_detail_denoise,
            )
            sess["poses"][i] = {
                "seed": seeds[i], "prompt": prompts[i], "image": image, "mask": mask,
                "edit_instruction": instruction, "edit_source_image": source_image,
            }
            comfy.model_management.soft_empty_cache()

        def apply_edit(i, instruction):
            """[2026-09-20] A one-off targeted edit of a pose's OWN already-
            generated pixels (e.g. "add high heel shoes") - not a fresh
            regenerate from the guide/character references. Runs before
            to_generate is computed, so the edited result is already in
            sess["poses"] by the time that's built - since seed/prompt for
            this pose are left unchanged, the normal to_generate equality
            check naturally leaves it alone afterward, no special exclusion
            needed. Returns True if it actually ran. Shared by both the
            single-pose "edit" action and "edit_all".
            """
            instruction = (instruction or "").strip()
            if confirmed[i] or not instruction:
                return False
            if sess["poses"].get(i) is None:
                # Best-effort recovery from the pose's own last saved preview
                # file - same reasoning as the confirmed-pose restore above,
                # but non-fatal: an edit request right after a restart (or a
                # settings change that wiped this unconfirmed pose) should
                # try to recover the pixels rather than silently falling
                # through to a normal fresh regeneration that discards the
                # edit instruction with zero feedback.
                pixels = _load_preview_pixels(state, i)
                if pixels is not None:
                    sess["poses"][i] = {"seed": seeds[i], "prompt": prompts[i], "image": pixels, "mask": None}
            if sess["poses"].get(i) is None:
                return False
            pose = sess["poses"][i]
            prior_instruction = pose.get("edit_instruction")
            if prior_instruction:
                # [2026-09-22] Anchor off the SAME pristine pre-edit pixels
                # every time (edit_source_image), not this edit's own output -
                # this is a full denoise=1.0 regeneration guided to resemble
                # its source, not a touch-up, so repeatedly feeding it its own
                # prior output compounds like a photocopy of a photocopy
                # (visible as color drift after 3-4 stacked edits). Folding
                # the new instruction in alongside the old one keeps this a
                # SINGLE full regeneration off a clean source instead of a
                # chain of them.
                base_image = pose["edit_source_image"]
                combined_instruction = f"{prior_instruction}; {instruction}"
            else:
                base_image = pose["image"]
                combined_instruction = instruction
            pose_name = POSE_NAMES[i]
            print(f"[MuseCharacterSheetDirector] editing {pose_name}: {combined_instruction}", flush=True)
            run_edit(i, base_image, combined_instruction)
            return True

        def reroll_edit(i, new_seed):
            """[2026-09-20] "New seed" on a pose that currently has an active
            edit re-runs THAT SAME edit instruction with a new seed, sourced
            from edit_source_image (the pixels the edit was originally
            applied to) - not the base pose, and not the previous edit
            result. Without this, the only "New seed" available reverted to
            the un-edited pose every time, discarding the edit entirely -
            exactly the behavior Andy flagged. Returns True if it actually
            re-rolled; False means "nothing to reroll here" (no active edit
            tracked, most likely because a ComfyUI restart wiped the
            in-process session - edit_source_image has no disk-backed
            recovery the way confirmed-pose pixels do), and the caller falls
            back to a normal base reroll instead of silently doing nothing.
            """
            if confirmed[i]:
                return False
            pose = sess["poses"].get(i)
            instruction = pose.get("edit_instruction") if pose else None
            source_image = pose.get("edit_source_image") if pose else None
            if not instruction or source_image is None:
                return False
            seeds[i] = int(new_seed)
            pose_name = POSE_NAMES[i]
            print(f"[MuseCharacterSheetDirector] re-rolling edit on {pose_name}: {instruction} (seed={seeds[i]})", flush=True)
            run_edit(i, source_image, instruction)
            return True

        if action and action.get("type") == "edit":
            apply_edit(int(action["pose"]), action.get("instruction"))
        elif action and action.get("type") == "edit_all":
            # [2026-09-20] Same edit, applied to every UNCONFIRMED pose in one
            # go - confirmed/locked poses are silently skipped, same lock
            # semantics as everything else in this node.
            instruction = action.get("instruction")
            for i in range(5):
                apply_edit(i, instruction)
        elif action and action.get("type") == "reroll_edit":
            i = int(action["pose"])
            if not reroll_edit(i, action["seed"]):
                # Nothing to reroll (see reroll_edit's docstring) - fall back
                # to a normal base reroll rather than doing nothing at all.
                if not confirmed[i]:
                    seeds[i] = int(action["seed"])
        elif action and action.get("type") == "reset_edit":
            # [2026-09-17] Cancels an active edit and reverts to the plain
            # base pose - ported from the Klein sibling node's identical
            # feature (a YouTube comment: edit an outfit, decide you don't
            # want it, but there was no way back to the un-edited version
            # short of retyping the base prompt). Just dropping the cached
            # pose is enough: seed/prompt haven't changed, so to_generate's
            # mismatch check won't catch it on its own, but with no cached
            # entry at all it unconditionally regenerates at the CURRENT
            # seed/prompt - a plain base generation, since sess["poses"][i]
            # never had edit_instruction/edit_source_image keys to begin
            # with once rebuilt this way.
            i = int(action["pose"])
            if not confirmed[i]:
                sess["poses"].pop(i, None)
        elif action and action.get("type") == "reset_prompt":
            # [2026-09-17] "Reset prompt to default" - Andy: "even if you
            # change it and mess it all up, you can hit default prompt and
            # it will do that." Reverts this pose's prompt back to its
            # built-in POSE_PROMPTS text and lets it regenerate with it -
            # the normal to_generate mismatch check (prompt changed) picks
            # this up on its own, no special regeneration path needed.
            i = int(action["pose"])
            if not confirmed[i]:
                prompts[i] = DEFAULT_PROMPTS[i]
        elif not action and seed_mode == "random":
            # [2026-09-23] A bare "hit Run" (no button clicked - action is
            # None) used to just replay whatever was already cached, since
            # nothing about seeds/prompts/settings had changed - correct for
            # avoiding wasted GPU work, but not what Andy actually wants:
            # a normal ComfyUI workflow with a random seed regenerates every
            # single time you queue it, full stop, no extra clicks required.
            # This reroll-everything-unconfirmed-on-a-bare-run is what makes
            # that true here too, while "fixed" mode (or a confirmed pose)
            # keeps the old reproducible/locked behavior untouched. Any
            # EXPLICIT action (reroll/edit/reroll_edit/finalize) already sets
            # its own seed(s) above and is excluded by the `not action` check,
            # so this can't double-randomize a pose the user just picked.
            for i in range(5):
                if not confirmed[i]:
                    seeds[i] = random.randint(0, 2 ** 31 - 1)

        to_generate = [i for i in range(5)
                       if not confirmed[i] and (
                           sess["poses"].get(i) is None
                           or sess["poses"][i]["seed"] != seeds[i]
                           or sess["poses"][i]["prompt"] != prompts[i]
                       )]
        for i in to_generate:
            print(f"[MuseCharacterSheetDirector] generating {POSE_NAMES[i]} (seed={seeds[i]})", flush=True)
            image, mask = _generate_pose(i, guide_image, character_image, sess["char_latent"], models,
                                          seeds[i], prompts[i], ref_boost, ref_boost_a, steps, cfg,
                                          face_detail, face_detail_type, face_detail_sampler,
                                          face_detail_scheduler, face_detail_denoise)
            sess["poses"][i] = {"seed": seeds[i], "prompt": prompts[i], "image": image, "mask": mask}
            # [2026-09-18] Reclaims VRAM fragmented/left over from this pose's
            # sampling+decode+RMBG+face-detail activations. Deliberately NOT
            # comfy.model_management.unload_all_models() - that would also
            # evict the cached UNET/CLIP/VAE this loop is reusing across all 5
            # poses, forcing a reload (and its cost) on the very next pose.
            # soft_empty_cache() only clears the CUDA allocator's free-but-held
            # blocks; it doesn't touch what's still actually in use.
            comfy.model_management.soft_empty_cache()

        want_finalize = bool(action) and action.get("type") == "finalize"
        final_image = None
        if want_finalize:
            if not all(confirmed):
                status = "not_all_confirmed"
            else:
                print("[MuseCharacterSheetDirector] assembling final sheet", flush=True)
                final_image = _assemble_final(sess, models, OUTPUT_SIZE_PRESETS.get(output_size, 1.0))
                status = "finalized"
        else:
            status = "generating" if to_generate else "ready"

        # [2026-09-19] ComfyUI's own execution.py aggregates a node's "ui" dict
        # assuming EVERY value is a list (it does `for y in x[k]` on each one -
        # see get_output_from_returns). pose_labels/seeds/prompts/confirmed are
        # genuinely 5-element lists already, so they were always fine. Scalar
        # fields have to be wrapped in a single-element list even though
        # there's only one value - status and final_preview were NOT wrapped,
        # which didn't crash (a str/dict is iterable) but silently corrupted
        # them: status got exploded into individual characters, and
        # final_preview got exploded into its own key names
        # (["filename","subfolder","type"]) instead of the real image
        # reference - almost certainly the actual cause of the "broken final
        # image" seen earlier, not a stale file. "reset" (added last turn) IS
        # a bool, which isn't iterable at all, hence the hard crash.
        pose_previews = [_preview(sess["poses"][i]["image"], persist=confirmed[i]) for i in range(5)]
        ui = {
            "pose_previews": pose_previews,
            "pose_labels": POSE_LABELS,
            "seeds": seeds,
            "prompts": prompts,
            "confirmed": confirmed,
            "status": [status],
        }
        if final_image is not None:
            # [2026-09-19] No in-node "final sheet" preview at all anymore -
            # the node's actual character_sheet output already goes wherever
            # the user's own SaveImage node puts it, so a second preview here
            # was just redundant screen space (and had briefly been a
            # redundant SAVED FILE too, via a persist=True mistake, before
            # that was reverted). Nothing to build/send for it.
            # A finished sheet is a completed job, not an
            # in-progress one - reset the session and the confirm/seed locks so
            # hitting Queue Prompt again (e.g. after swapping in a different
            # character/guide image) starts a fresh sheet instead of instantly
            # re-finalizing the same one, or inheriting confirmed-pose locks
            # that no longer make sense once a new image is in the inputs.
            # pose_previews above was already built from the just-finalized
            # poses, so the completed sheet/poses still display; only the
            # locks and next run's starting point are reset. "reset": True
            # tells the JS side to overwrite its own locally-held
            # confirmed/seeds state to match (it otherwise treats confirmed as
            # client-owned, to dodge a race - see update() in the JS).
            sess.clear()
            # [2026-09-20] seed_mode governs ONLY this post-finalize reset -
            # per-pose "New seed" always picks a fresh random seed regardless
            # of this setting (see the reroll action, unchanged). Without a
            # "random" option here, running the SAME guide/character photos
            # again after a completed sheet would always restart from the
            # exact same DEFAULT_SEEDS and produce byte-identical poses - the
            # only way to get a different result would be feeding in
            # different images, which isn't what "just click Run again"
            # should require.
            if seed_mode == "random":
                ui["seeds"] = [random.randint(0, 2 ** 31 - 1) for _ in range(5)]
            else:
                ui["seeds"] = list(DEFAULT_SEEDS)
            ui["confirmed"] = [False] * 5
            ui["reset"] = [True]

        # Silent block (message=None): "not finalized yet" is the normal, expected
        # state while the user is still reviewing/re-rolling poses - it should not
        # surface as a red error every time a downstream SaveImage runs before the
        # sheet is confirmed.
        result = (final_image,) if final_image is not None else (ExecutionBlocker(None),)
        return {"ui": ui, "result": result}


NODE_CLASS_MAPPINGS = {"MuseCharacterSheetDirector": MuseCharacterSheetDirector}
# [2026-09-19] Display name only - "Director" dropped per Andy's request
# (the H3 Director reference was just to point at that node's confirm/reroll
# UI pattern, not a naming cue). The registered key above and the class name
# stay MuseCharacterSheetDirector on purpose - renaming those would break the
# class_type reference already saved in the workflow that's now working.
NODE_DISPLAY_NAME_MAPPINGS = {"MuseCharacterSheetDirector": "Muse Character Sheet"}
