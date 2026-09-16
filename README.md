# Muse Character Sheet

[Krea 2](https://huggingface.co/krea/Krea-2-Turbo) is Krea AI's instruction-based image editing model. **Muse Character Sheet** is a single ComfyUI node that generates a full 5-pose character turnaround (portrait, front, left profile, right profile, back) from one mannequin pose guide + one character reference photo, using Krea 2 Edit with identity-preserving LoRAs.

Instead of wiring up five separate Krea2Edit chains by hand, the node runs all five internally and gives you a **Confirm** / **New seed** button per pose in its own panel. Confirm is a pure local lock — it doesn't submit anything on its own. Once every pose is confirmed, click **Build final sheet now** to assemble the final 4096x2304 sheet.

Since Krea2Edit is an edit model, each pose also gets an **Apply edit** box for targeted fixes (e.g. "add high heel shoes") without a full regenerate, plus an **Edit all poses** box to apply one instruction to every unconfirmed pose at once. "New seed" on a pose with an active edit re-rolls *that edit*, not the original. A **seed_mode** widget (`random`/`fixed`) controls whether re-running the same photos after a completed sheet gives you a fresh random set or a reproducible one.

## ⚠️ Required custom nodes — install these BEFORE you run anything

**ComfyUI's own "Install Missing Custom Nodes" may not catch all of these** — some are used internally by the node's own code, not as separate nodes on the canvas.

- **[comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)** — image-grounded instruction encoding for Krea 2 (`Krea2EditGroundedEncode`). Required unconditionally.
- **[ComfyUI-Krea2-NAG](https://github.com/iljung1106/ComfyUI-Krea2-NAG)** — Normalized Attention Guidance for Krea2Edit (`Krea2EditNormalizedAttentionGuidance`). Required unconditionally.
- **[ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG)** — required for the white-background cleanup run on every pose. The RMBG-2.0 model it uses auto-downloads on first use.
- **[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** + **[ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack)** — only required if you enable the node's **Face Detail** pass (off by default).

This repo also bundles **Muse Sheet: Align Figure Height** (`MuseSheetAlignFigure`), used internally to size/align every panel in the final sheet — no separate install needed.

## Model Links

### Diffusion model (used by the `unet_name` widget)

[🤗 Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)

**diffusion_models**
- [krea2_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_int8_convrot.safetensors) (13.5 GB)

### Text encoder (`clip_name`, type `krea2`)

- [qwen3vl_4b_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) (5.24 GB) — same repo as above, `text_encoders/` subfolder

### VAE (`vae_name`)

- [qwen_image_vae.safetensors](https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors) (254 MB) — same repo, `vae/` subfolder

### LoRAs

[🤗 conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit)
- [krea2_identity_edit_v1_2.safetensors](https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors) — `identity_lora` widget. The official Identity Edit LoRA behind `comfyui-krea2edit`.

[🤗 ivanlf98/DetailerKrea](https://huggingface.co/ivanlf98/DetailerKrea)
- [Detailer-KREA2.safetensors](https://huggingface.co/ivanlf98/DetailerKrea/resolve/main/Detailer-KREA2.safetensors) — used as `lora_3` in the Muse Model Loader instance feeding this node.

[🤗 uzumix/krea2filterbypass3.safetensors](https://huggingface.co/uzumix/krea2filterbypass3.safetensors)
- [krea2filterbypass3.safetensors](https://huggingface.co/uzumix/krea2filterbypass3.safetensors/resolve/main/krea2filterbypass3.safetensors) — `filter_bypass_lora` widget (save it locally as `krea2filterbypass3_fp32.safetensors`, or just keep the downloaded filename and point the widget at whatever you call it).

## Model Storage Locations

- `ComfyUI/models/diffusion_models/` — `krea2_turbo_int8_convrot.safetensors`
- `ComfyUI/models/text_encoders/` — `qwen3vl_4b_fp8_scaled.safetensors`
- `ComfyUI/models/vae/` — `qwen_image_vae.safetensors`
- `ComfyUI/models/loras/Krea2/` — `krea2_identity_edit_v1_2.safetensors`, `krea2filterbypass3_fp32.safetensors`, `Detailer-KREA2.safetensors`
- `ComfyUI/models/ultralytics/bbox/` — `face_yolov8m.pt` *(only needed if you enable Face Detail; ships/auto-downloads with Impact-Subpack)*

## Links
- [Krea 2 on Hugging Face](https://huggingface.co/krea/Krea-2-Turbo)
- [🤗 Comfy-Org/Krea-2 (ComfyUI-ready weights)](https://huggingface.co/Comfy-Org/Krea-2)
- [Muse Character Sheet on GitHub](https://github.com/muse-collective-26/muse-character-sheet)
- [Muse Model Loader on GitHub](https://github.com/muse-collective-26/muse-model-loader)
