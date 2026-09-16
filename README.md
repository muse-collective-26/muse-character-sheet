# Muse Character Sheet

[2026-09-15] Single ComfyUI node that wraps the "Krea2 Character Sheet - Aligned
Views" workflow (`Muse Collective Krea2 Character Sheet - Aligned Views.json`)
into one interactive UI, modelled on the same confirm/re-roll pattern used by
`Muse-MiniMax-Seed-Hunt-Studio` and the Director nodes in this custom_nodes
folder.

## What it does

- Two required image inputs: `guide_image` (the 5-panel mannequin pose guide
  sheet) and `character_image` (a single reference photo of the person/outfit).
- Three optional MODEL/CLIP/VAE sockets (`model_override`/`clip_override`/
  `vae_override`) - leave them unconnected and the node loads its own model
  from the `unet_name`/`identity_lora`/`filter_bypass_lora`/`clip_name`/
  `vae_name` widgets as before. Connect them (e.g. from a
  [Muse Model Loader](https://github.com/) instance) and they override the
  widgets - any LoRAs already baked into an incoming model travel with it, so
  the node skips applying `identity_lora`/`filter_bypass_lora` itself in that
  case. (e.g. from a "Muse Model Loader" node elsewhere in this custom_nodes
  tree, or any other node that outputs MODEL/CLIP/VAE.)
- On first queue it generates all five poses (portrait close-up, front, left
  profile, right profile, back) with the same Krea2 models/LoRAs/prompts as the
  original workflow.
- The node's own panel shows all five previews with a **Confirm** and a
  **New seed** (re-roll) button per pose.
  - Re-roll only regenerates that one pose (new random seed), leaving the
    other four untouched.
  - Confirm is a pure local toggle - it locks that pose (disabling re-roll/
    prompt-editing on it) but does **not** submit anything on its own, not
    even on the 5th/last pose. The only thing that ever builds the sheet is
    clicking **Build final sheet now**.
- Output: the assembled 4096x2304 character sheet (`character_sheet`), blocked
  (silently, not as an error) until every pose is confirmed and Build final
  sheet has actually been clicked.
- A completed sheet resets the node's confirm/seed locks automatically, so the
  next Queue Prompt (e.g. after swapping in a different character/guide photo)
  starts a fresh sheet instead of instantly re-finalizing the old one.
- Confirmed poses persist to your ComfyUI output folder (not the ephemeral
  temp folder), so a confirmed lock survives a ComfyUI restart.

## Why it works the way it does

ComfyUI has no live pause/resume inside a single node execution - Python can
only act from inside a real `/prompt` submission. So every button click that
needs new pixels (re-roll, apply edited prompt, build final sheet) fires its
own small `app.queuePrompt()` call from the JS side; state that must survive
between those calls (which poses are generated, their seeds, confirm flags)
rides in a hidden `state_json` widget serialized with the node, plus an
in-process cache in `character_sheet_director.py` keyed by the node's
`unique_id`. Confirm itself is deliberately the one action that does NOT
queue anything - it's a pure client-side lock toggle.

## One deliberate deviation from the source workflow JSON

The source workflow chains `Krea2EditModelPatch` -> `Krea2NormalizedAttentionGuidance`
as two separate nodes. The currently installed `krea2-nag` package's plain
`Krea2NormalizedAttentionGuidance` now raises `ValueError` if it's applied to a
model that already has the `krea2_edit` wrapper (i.e. already patched by
`Krea2EditModelPatch`) - it tells you to use the combined node instead. This
node therefore calls `Krea2EditNormalizedAttentionGuidance` (the combined
edit-patch + NAG node) directly on the base LoRA model, passing the same
source_latent/source_image/ref_boost/target_latent arguments that the old
workflow split across two nodes. Functionally equivalent (same docstring:
"source-reference attention stays on the positive path; NAG is applied only to
target image tokens"), but it's the only wiring the currently installed
package will actually run without erroring.

## Model widgets

`unet_name` / `identity_lora` / `filter_bypass_lora` / `clip_name` / `vae_name`
default to whatever is first in your installed lists - pick the same files the
source workflow used, or leave them alone and connect a model loader to the
optional override sockets instead:
- unet: `krea2_turbo_int8_convrot.safetensors`
- identity_lora: `Krea2\\krea2_identity_edit_v1_2.safetensors`
- filter_bypass_lora: `Krea2\\krea2filterbypass3_fp32.safetensors`
- clip: `qwen3vl_4b_fp8_scaled.safetensors` (type `krea2`)
- vae: `qwen_image_vae.safetensors`

`ref_boost` (default 2.0) / `ref_boost_a` (default 4.0) match the source
workflow's actual runtime values (the outer `Seed`/`JWIntegerToFloat` nodes it
fed into `Krea2EditModelPatch` - not the 10/3 defaults baked into the widget,
which were overridden by those links).

## Guide-sheet crop rects

The five pose crop rectangles are tuned to the original 1670x942 mannequin
guide layout and scale proportionally if you feed in a differently-sized guide
image - but they assume the same panel ORDER and proportions (portrait, front,
right-profile, left-profile, back, left-to-right in the guide sheet). A
differently laid-out guide sheet will need new crop rects in `GUIDE_CROPS` in
`character_sheet_director.py`.
