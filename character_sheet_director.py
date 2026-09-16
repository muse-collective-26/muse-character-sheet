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
# size; the four full-body views share the aligned-figure canvas size.
TARGET_SIZE = {
    "01_portrait": (544, 976),
    "02_front": (544, 1784),
    "03_left_profile": (544, 1784),
    "04_right_profile": (544, 1784),
    "05_back": (544, 1784),
}

ALIGN_GUIDE_KW = dict(width=544, height=1784, figure_height=1584, bottom_margin=100, threshold=0.1)
ALIGN_BODY_KW = dict(width=704, height=2304, figure_height=2048, bottom_margin=128, threshold=0.1)
PORTRAIT_FINAL_SIZE = (1280, 2304)  # (width, height)

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
FACE_DETAIL_FIXED_KW = dict(
    guide_size=1536.0, guide_size_for=False, max_size=1536.0,  # guide_size_for False = "crop_region"
    feather=10, noise_mask=True, force_inpaint=True,
    bbox_threshold=0.50, bbox_dilation=10, bbox_crop_factor=2.0,
    sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93,
    sam_bbox_expansion=0, sam_mask_hint_threshold=0.70, sam_mask_hint_use_negative="False",
    drop_size=10, wildcard="", cycle=1,
    inpaint_model=False, noise_mask_feather=10, tiled_encode=False, tiled_decode=False,
)

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
        # [2026-09-19] Overrides are live MODEL/CLIP/VAE objects, not filenames -
        # id() changes whenever the upstream loader (e.g. Muse Model Loader)
        # actually re-executes and hands back a new object, which is exactly
        # the signal needed to invalidate a stale session. Connecting/
        # disconnecting a socket also changes None<->int here, so switching
        # between manual widgets and an override correctly resets the session.
        id(model_override) if model_override is not None else None,
        id(clip_override) if clip_override is not None else None,
        id(vae_override) if vae_override is not None else None,
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
    detector = _get_detector(face_detail_type)
    print(f"[MuseCharacterSheetDirector] >>> face-detail pass START (detect={face_detail_type}, "
          f"sampler={face_detail_sampler}, scheduler={face_detail_scheduler}, "
          f"denoise={face_detail_denoise}, steps={FACE_DETAIL_STEPS})", flush=True)
    result = _node("FaceDetailer").doit(
        image=image, model=nag_model, clip=models["clip"], vae=models["vae"],
        seed=int(seed), steps=FACE_DETAIL_STEPS, cfg=float(cfg), sampler_name=face_detail_sampler,
        scheduler=face_detail_scheduler, positive=positive, negative=negative,
        denoise=float(face_detail_denoise), bbox_detector=detector["bbox_detector"],
        **FACE_DETAIL_FIXED_KW,
    )
    mask = result[3]
    found = bool(mask is not None and mask.numel() > 0 and mask.any())
    print(f"[MuseCharacterSheetDirector] <<< face-detail pass END - "
          f"{'region found and refined' if found else 'NO region detected, image unchanged'}", flush=True)
    return result[0]


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
    import os
    import numpy as np
    from PIL import Image
    previews = state.get("previews") or []
    item = previews[index] if index < len(previews) else None
    if not item or item.get("type") not in ("temp", "output"):
        raise ValueError(
            f"Confirmed pose {index + 1} has no saved preview to restore from. "
            f"Unconfirm it to regenerate, or confirm it again once it's been generated."
        )
    base = folder_paths.get_temp_directory() if item["type"] == "temp" else folder_paths.get_output_directory()
    base = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base, item.get("subfolder", ""), item.get("filename", "")))
    if os.path.commonpath([base, path]) != base or not os.path.isfile(path):
        raise ValueError(
            f"Confirmed pose {index + 1}'s saved preview file is missing on disk "
            f"(temp previews can get cleaned up over time). Unconfirm it to regenerate."
        )
    with Image.open(path) as im:
        pixels = torch.from_numpy(np.array(im.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
    # The saved image already had RMBG applied (white background) before it was
    # written - re-run RMBG here just to recover a matching foreground mask,
    # not to redo the background cleanup itself.
    _, mask, _ = _node("RMBG").process_image(image=pixels, **RMBG_KW)
    return {"seed": seed, "prompt": prompt, "image": pixels, "mask": mask}


def _assemble_final(sess, models):
    panels = []
    for i in range(5):
        pose = sess["poses"][i]
        if i == 0:
            panel = _resize(pose["image"], *PORTRAIT_FINAL_SIZE)
        else:
            panel = _node("MuseSheetAlignFigure").align(image=pose["image"], mask=pose["mask"], **ALIGN_BODY_KW)[0]
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
                "state_json": ("STRING", {"multiline": True, "default": json.dumps(DEFAULT_STATE)}),
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
            state_json, unique_id, model_override=None, clip_override=None, vae_override=None):
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
                final_image = _assemble_final(sess, models)
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
