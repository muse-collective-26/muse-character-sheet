"""Non-generative, aspect-preserving character-sheet alignment. No model patches.

[2026-09-23] Bundled directly into this package (previously lived in a
local-only, never-published ComfyUI-Muse-Sheet-Alignment folder) so this repo
is self-contained - no separate custom node install required to line up the
final sheet's panels. The sibling Klein repo bundles an identical copy of
this same file for the same reason; loading both packages registers
"MuseSheetAlignFigure" twice (ComfyUI logs a harmless duplicate-registration
warning, one copy just wins - the code is byte-identical either way).
"""
import logging
import torch
import torch.nn.functional as F


class MuseSheetAlignFigure:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 704, "min": 16, "max": 8192}),
            "height": ("INT", {"default": 2304, "min": 16, "max": 8192}),
            "figure_height": ("INT", {"default": 2048, "min": 16, "max": 8192}),
            "bottom_margin": ("INT", {"default": 128, "min": 0, "max": 8192}),
            "threshold": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 0.99, "step": 0.01}),
        }, "optional": {"mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align"
    CATEGORY = "Muse/Character Sheet"

    def align(self, image, width, height, figure_height, bottom_margin, threshold=0.1, mask=None):
        if figure_height + bottom_margin > height:
            raise ValueError("Figure height plus bottom margin must fit inside the canvas.")
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Expected an RGB IMAGE batch.")
        if mask is None:
            # White-background mannequin guides only. Generated views use RMBG masks.
            mask = (1.0 - image.min(dim=-1).values).clamp(0, 1)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] not in (1, image.shape[0]):
            raise ValueError("Mask batch must contain one mask or one per image.")
        mask = mask.to(device=image.device, dtype=torch.float32)
        if mask.shape[-2:] != image.shape[1:3]:
            mask = F.interpolate(mask[:, None], size=image.shape[1:3], mode="bilinear", align_corners=False)[:, 0]
        result = []
        for i, frame in enumerate(image):
            foreground = mask[min(i, mask.shape[0]-1)] > threshold
            # Ignore isolated one-pixel mask noise, retaining disconnected shoes/hair.
            rows = torch.where(foreground.sum(dim=1) >= 3)[0]
            cols = torch.where(foreground.sum(dim=0) >= 3)[0]
            if not len(rows) or not len(cols):
                raise ValueError("No subject detected. Check the foreground mask or white guide background.")
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            scaled_width = max(1, round((x1-x0) * figure_height / (y1-y0)))
            if scaled_width > width:
                raise ValueError(f"Subject needs a {scaled_width}px wide panel at this height, but panel is {width}px. "
                                 "Increase all body-panel widths or lower all figure heights equally; do not squash the figure.")
            crop = frame[y0:y1, x0:x1].permute(2, 0, 1)[None].float()
            resized = F.interpolate(crop, size=(figure_height, scaled_width), mode="bicubic", align_corners=False, antialias=True)
            resized = resized[0].permute(1, 2, 0).clamp(0, 1).to(frame.dtype)
            canvas = torch.ones((height, width, 3), device=frame.device, dtype=frame.dtype)
            top = height - bottom_margin - figure_height
            left = (width - scaled_width) // 2
            canvas[top:top+figure_height, left:left+scaled_width] = resized
            logging.info("[Muse Sheet Alignment] bounds=%s -> figure=%sx%s canvas=%sx%s bottom=%s",
                         (x0,y0,x1,y1), scaled_width,figure_height,width,height,bottom_margin)
            result.append(canvas)
        return (torch.stack(result),)


NODE_CLASS_MAPPINGS = {"MuseSheetAlignFigure": MuseSheetAlignFigure}
NODE_DISPLAY_NAME_MAPPINGS = {"MuseSheetAlignFigure": "Muse Sheet: Align Figure Height"}
