const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const POSE_LABELS = ["Portrait (close-up)", "Front", "Left profile", "Right profile", "Back"];
const POSE_ASPECT = ["544 / 976", "544 / 1784", "544 / 1784", "544 / 1784", "544 / 1784"];
const DEFAULT_SEEDS = [41001, 41002, 41003, 41004, 41005];

function defaultState() {
  return {
    seeds: [...DEFAULT_SEEDS],
    confirmed: [false, false, false, false, false],
    prompts: ["", "", "", "", ""],
    previews: [null, null, null, null, null],
    status: null,
    action: null,
  };
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.options ??= {};
  widget.options.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  if (widget.element) widget.element.style.display = "none";
}

// Plain `widget.value = x` never fires the widget's own callback, which is
// what tells ComfyUI the graph changed and needs re-saving. Skipping that is
// why a toggle flipped via our custom controls could silently revert: switch
// workflow tabs and back, and ComfyUI restores the last state that WAS
// properly captured - before our change.
function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value);
}

// [2026-09-19] Purely cosmetic feedback for the optional model/clip/vae
// override sockets: when one is connected, its matching manual widget(s) are
// dimmed and disabled to show they're being ignored (Python's _get_models()
// is what actually enforces the override - this just keeps the UI honest).
// These stay real native LiteGraph widgets, not DOM elements, so none of the
// addDOMWidget/ResizeObserver/border-radius issues from the rest of this
// node's UI apply here.
const OVERRIDE_WIDGET_MAP = {
  model_override: ["unet_name", "identity_lora", "filter_bypass_lora"],
  clip_override: ["clip_name"],
  vae_override: ["vae_name"],
};

function applyOverrideState(node) {
  for (const [inputName, widgetNames] of Object.entries(OVERRIDE_WIDGET_MAP)) {
    const input = (node.inputs || []).find((inp) => inp.name === inputName);
    const connected = !!input?.link;
    for (const wName of widgetNames) {
      const widget = (node.widgets || []).find((w) => w.name === wName);
      if (!widget) continue;
      widget.disabled = connected;
      widget.options ??= {};
      widget.options.disabled = connected;
    }
  }
  node.setDirtyCanvas?.(true, true);
}

function enableCanvasZoomOverDOM(root) {
  root.addEventListener(
    "wheel",
    (event) => {
      const target = event.target;
      const interactive = target?.closest?.("textarea, select, input, button, [contenteditable='true']");
      if (interactive) return;
      const canvas = app?.canvas?.canvas;
      if (!canvas) return;
      event.preventDefault();
      event.stopPropagation();
      canvas.dispatchEvent(
        new WheelEvent("wheel", {
          deltaX: event.deltaX,
          deltaY: event.deltaY,
          deltaZ: event.deltaZ,
          deltaMode: event.deltaMode,
          clientX: event.clientX,
          clientY: event.clientY,
          bubbles: true,
          cancelable: true,
        })
      );
    },
    { passive: false }
  );
}

function getState(stateWidget) {
  try {
    const parsed = JSON.parse(stateWidget.value || "{}");
    return {
      seeds: Array.isArray(parsed.seeds) && parsed.seeds.length === 5 ? parsed.seeds : [...DEFAULT_SEEDS],
      confirmed:
        Array.isArray(parsed.confirmed) && parsed.confirmed.length === 5
          ? parsed.confirmed
          : [false, false, false, false, false],
      prompts:
        Array.isArray(parsed.prompts) && parsed.prompts.length === 5
          ? parsed.prompts
          : ["", "", "", "", ""],
      previews:
        Array.isArray(parsed.previews) && parsed.previews.length === 5
          ? parsed.previews
          : [null, null, null, null, null],
      status: parsed.status || null,
      action: parsed.action || null,
    };
  } catch (e) {
    return defaultState();
  }
}

function setState(stateWidget, state) {
  setWidgetValue(stateWidget, JSON.stringify(state));
}

function describeStatus(status) {
  switch (status) {
    case "generating":
      return "Generating poses…";
    case "ready":
      return "All poses generated — review, confirm or re-roll each one.";
    case "not_all_confirmed":
      return "Confirm every pose before building the final sheet.";
    case "finalized":
      return "Done — character sheet ready below.";
    default:
      return "Queue the node to generate all five poses.";
  }
}

function populateSelect(select, widget, fallbackOptions) {
  const options = widget?.options?.values || fallbackOptions || [];
  select.innerHTML = "";
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    select.appendChild(el);
  }
  if (widget) select.value = widget.value;
}

function buildFaceDetailBox(faceWidgets) {
  const { faceDetail, faceDetailType, faceDetailSampler, faceDetailScheduler, faceDetailDenoise } = faceWidgets;

  const box = document.createElement("div");
  box.style.cssText =
    "display:flex;flex-direction:column;gap:6px;padding:8px;background:#111114;" +
    "border:1px solid #3f3f46;border-radius:6px;box-sizing:border-box;";

  const headerRow = document.createElement("label");
  headerRow.style.cssText = "display:flex;align-items:center;gap:6px;font-weight:600;font-size:12px;color:#d4d4d8;cursor:pointer;";
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.checked = !!faceDetail?.value;
  toggle.style.cssText = "width:14px;height:14px;cursor:pointer;";
  const headerText = document.createElement("span");
  headerText.textContent = "Face Detail";
  headerRow.appendChild(toggle);
  headerRow.appendChild(headerText);
  box.appendChild(headerRow);

  const fieldsRow = document.createElement("div");
  fieldsRow.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:6px;";
  box.appendChild(fieldsRow);

  function makeField(labelText) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:2px;";
    const label = document.createElement("div");
    label.textContent = labelText;
    label.style.cssText = "font-size:9px;color:#71717a;text-transform:uppercase;letter-spacing:0.03em;";
    wrap.appendChild(label);
    fieldsRow.appendChild(wrap);
    return wrap;
  }

  const selectStyle =
    "width:100%;padding:3px 4px;border-radius:4px;border:1px solid #3f3f46;background:#27272a;color:#e4e4e7;font-size:11px;box-sizing:border-box;";

  const detectWrap = makeField("Detect");
  const detectSelect = document.createElement("select");
  detectSelect.style.cssText = selectStyle;
  populateSelect(detectSelect, faceDetailType, ["face", "hand", "person"]);
  detectWrap.appendChild(detectSelect);

  const samplerWrap = makeField("Sampler");
  const samplerSelect = document.createElement("select");
  samplerSelect.style.cssText = selectStyle;
  populateSelect(samplerSelect, faceDetailSampler);
  samplerWrap.appendChild(samplerSelect);

  const schedulerWrap = makeField("Scheduler");
  const schedulerSelect = document.createElement("select");
  schedulerSelect.style.cssText = selectStyle;
  populateSelect(schedulerSelect, faceDetailScheduler);
  schedulerWrap.appendChild(schedulerSelect);

  const denoiseWrap = makeField("Denoise");
  const denoiseInput = document.createElement("input");
  denoiseInput.type = "number";
  denoiseInput.min = "0";
  denoiseInput.max = "1";
  denoiseInput.step = "0.01";
  denoiseInput.value = faceDetailDenoise ? faceDetailDenoise.value : 0.3;
  denoiseInput.style.cssText = selectStyle;
  denoiseWrap.appendChild(denoiseInput);

  function setEnabled(enabled) {
    for (const el of [detectSelect, samplerSelect, schedulerSelect, denoiseInput]) {
      el.disabled = !enabled;
      el.style.opacity = enabled ? "1" : "0.45";
    }
  }
  setEnabled(toggle.checked);

  toggle.onchange = () => {
    setWidgetValue(faceDetail, toggle.checked);
    setEnabled(toggle.checked);
  };
  detectSelect.onchange = () => setWidgetValue(faceDetailType, detectSelect.value);
  samplerSelect.onchange = () => setWidgetValue(faceDetailSampler, samplerSelect.value);
  schedulerSelect.onchange = () => setWidgetValue(faceDetailScheduler, schedulerSelect.value);
  denoiseInput.onchange = () => {
    const v = Math.min(1, Math.max(0, parseFloat(denoiseInput.value) || 0));
    denoiseInput.value = v;
    setWidgetValue(faceDetailDenoise, v);
  };

  // Re-reads the underlying widgets and syncs the visible controls. Needed
  // because these controls are only initialized once, at DOM-build time,
  // which happens before onConfigure loads a saved workflow's real widget
  // values - without this, the box would show fresh-node defaults after a
  // tab switch even though the widgets themselves hold the right values.
  function sync() {
    toggle.checked = !!faceDetail?.value;
    if (faceDetailType) detectSelect.value = faceDetailType.value;
    if (faceDetailSampler) samplerSelect.value = faceDetailSampler.value;
    if (faceDetailScheduler) schedulerSelect.value = faceDetailScheduler.value;
    if (faceDetailDenoise) denoiseInput.value = faceDetailDenoise.value;
    setEnabled(toggle.checked);
  }

  return { box, sync };
}

function buildCharacterSheetUI(node, stateWidget, faceWidgets) {
  const root = document.createElement("div");
  // No border-radius here, deliberately: this node has its own teal bgcolor
  // (set on the node itself, visible across the unet_name/ref_boost/etc. rows
  // above this box), and a rounded corner on a differently-coloured box always
  // shows a sliver of whatever's behind it in the cut-away curve, regardless
  // of how precisely the box is sized - that's what was showing at the
  // bottom-right corner. Flat corners have nothing to cut away, so there's no
  // sliver possible no matter what colour the node itself has.
  // [2026-09-19] min-height:100% REMOVED - it was the actual cause of both
  // the node starting absurdly oversized AND the button/grey mismatch never
  // going away no matter what applySize() did. It tied root's own rendered
  // height to its parent (the ComfyUI-managed .dom-widget wrapper)'s CURRENT
  // height. applySize() then measured root's height to decide how tall the
  // wrapper SHOULD be - a closed loop: grow the wrapper -> root's floor grows
  // to match -> that growth gets read back as "content needs more room" ->
  // grow the wrapper again. Every correction fed the next one, compounding
  // fast across the many resize ticks that fire during initial page load
  // (images loading, ResizeObserver re-firing) - which is exactly the
  // "starts absurdly tall" symptom, and made every height measurement
  // unreliable regardless of how applySize() computed the target (which is
  // also why the button/grey mismatch survived multiple different attempts
  // at the computation itself - the computation was never the real bug).
  // Root's height is now driven purely by its own content, with nothing
  // upstream feeding back into that number - applySize() below is the only
  // thing that resizes the wrapper, one direction, no loop.
  // [2026-09-19] Extra bottom padding (24px, vs 8px on the other 3 sides) is
  // a deliberate safety margin, not a layout requirement - every sizing bug
  // in this node so far has been the wrapper ending a few px SHORT of root's
  // real content, exposing the node's own rounded corner underneath in that
  // gap. Overshooting is invisible (it's just more of root's own background
  // colour); undershooting is what exposes the corner. Padding out the
  // bottom guarantees a buffer no residual measurement imprecision can eat
  // through, rather than continuing to chase pixel-perfect exactness.
  root.style.cssText =
    "display:flex;flex-direction:column;gap:8px;padding:8px 8px 24px 8px;background:#1b1b1f;" +
    "color:#e4e4e7;font-family:-apple-system,Segoe UI,sans-serif;font-size:12px;width:100%;" +
    "box-sizing:border-box;";

  const statusEl = document.createElement("div");
  statusEl.style.cssText = "font-weight:600;color:#9ca3af;min-height:16px;";
  statusEl.textContent = describeStatus(null);
  root.appendChild(statusEl);

  const faceDetailBox = buildFaceDetailBox(faceWidgets);
  root.appendChild(faceDetailBox.box);

  const grid = document.createElement("div");
  grid.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;";
  root.appendChild(grid);

  const panels = [];
  for (let i = 0; i < 5; i++) {
    const card = document.createElement("div");
    card.style.cssText =
      "flex:1 1 150px;min-width:140px;max-width:220px;display:flex;flex-direction:column;gap:4px;" +
      "background:#111114;border:2px solid #2c2c33;border-radius:6px;padding:6px;box-sizing:border-box;";

    const label = document.createElement("div");
    label.textContent = POSE_LABELS[i];
    label.style.cssText = "font-weight:600;font-size:11px;color:#d4d4d8;";
    card.appendChild(label);

    const imgWrap = document.createElement("div");
    imgWrap.style.cssText = `width:100%;aspect-ratio:${POSE_ASPECT[i]};background:#000;border-radius:4px;overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative;`;
    const img = document.createElement("img");
    img.style.cssText = "width:100%;height:100%;object-fit:contain;display:none;";
    imgWrap.appendChild(img);
    const placeholder = document.createElement("div");
    placeholder.textContent = "not generated yet";
    placeholder.style.cssText = "color:#52525b;font-size:10px;text-align:center;padding:4px;";
    imgWrap.appendChild(placeholder);
    card.appendChild(imgWrap);

    const seedLabel = document.createElement("div");
    seedLabel.style.cssText = "font-size:10px;color:#71717a;";
    seedLabel.textContent = `seed ${DEFAULT_SEEDS[i]}`;
    card.appendChild(seedLabel);

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:4px;";

    const confirmBtn = document.createElement("button");
    confirmBtn.textContent = "Confirm";
    confirmBtn.style.cssText =
      "flex:1;min-height:32px;padding:5px 3px;border-radius:4px;border:1px solid #22c55e;background:#166534;color:#ffffff;cursor:pointer;font-size:12px;";

    const rerollBtn = document.createElement("button");
    rerollBtn.textContent = "\u{1F3B2} New seed";
    rerollBtn.style.cssText =
      "flex:1;min-height:32px;padding:5px 3px;border-radius:4px;border:1px solid #60a5fa;background:#1d4ed8;color:#ffffff;cursor:pointer;font-size:12px;";

    btnRow.appendChild(confirmBtn);
    btnRow.appendChild(rerollBtn);
    card.appendChild(btnRow);

    const promptToggle = document.createElement("button");
    promptToggle.textContent = "Show prompt ▾";
    promptToggle.style.cssText =
      "padding:3px 2px;border-radius:4px;border:1px solid #2c2c33;background:transparent;color:#71717a;cursor:pointer;font-size:10px;";
    card.appendChild(promptToggle);

    const promptWrap = document.createElement("div");
    promptWrap.style.cssText = "display:none;flex-direction:column;gap:4px;";
    const promptArea = document.createElement("textarea");
    promptArea.style.cssText =
      "width:100%;min-height:110px;resize:vertical;padding:4px;border-radius:4px;border:1px solid #3f3f46;" +
      "background:#0b0b0d;color:#d4d4d8;font-size:10px;font-family:inherit;box-sizing:border-box;";
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply prompt (same seed)";
    applyBtn.style.cssText =
      "padding:4px 2px;border-radius:4px;border:1px solid #3f3f46;background:#27272a;color:#e4e4e7;cursor:pointer;font-size:11px;";
    promptWrap.appendChild(promptArea);
    promptWrap.appendChild(applyBtn);
    card.appendChild(promptWrap);

    grid.appendChild(card);
    panels.push({
      card, img, placeholder, seedLabel, confirmBtn, rerollBtn,
      promptToggle, promptWrap, promptArea, applyBtn,
    });
  }

  // [2026-09-19] No in-node "final sheet" preview any more - the node's real
  // character_sheet output already goes wherever the user's own SaveImage
  // node puts it, so a second preview here was pure redundant screen space
  // (and had briefly been a redundant SAVED FILE too, before that got
  // reverted). See run()'s ui dict in the Python - it no longer sends a
  // final_preview at all.

  const buildBtn = document.createElement("button");
  buildBtn.textContent = "Build final sheet now";
  buildBtn.style.cssText =
    "min-height:32px;padding:6px;border-radius:4px;border:1px solid #c2793f;background:#6b3a1a;color:#f4ddc4;cursor:pointer;font-size:12px;font-weight:600;";
  root.appendChild(buildBtn);

  function queue() {
    app.queuePrompt(0, 1);
  }

  function measureHeight() {
    return Math.max(240, Math.ceil(root.scrollHeight + 8));
  }

  function applySize() {
    // [2026-09-19] A delta-based version of this (node.size[1] + delta)
    // could only ever nudge the node's CURRENT size - it never fixed an
    // already-wrong size sitting in the saved workflow (e.g. left over from
    // the min-height:100% runaway-growth bug below), because once the
    // wrapper and root roughly agreed with each other, the delta came out
    // near zero and the stale oversized node.size[1] just stayed put. This
    // sets the total height ABSOLUTELY every time via node.computeSize() -
    // LiteGraph's own aggregate of header + every widget above the DOM box +
    // the DOM box's own height (domWidget.computeSize below is wired to
    // measureHeight()) - so it self-corrects any stale size immediately
    // rather than only reacting to future changes. This also fixes root's
    // OWN sizing bug: root used to have min-height:100%, which tied its
    // rendered height to its parent (the wrapper)'s CURRENT height - so
    // growing the wrapper grew root's floor to match, which fed back in as
    // "content needs more room", growing the wrapper again, compounding
    // across the many resize ticks that fire during page load. That's what
    // produced a node that started absurdly tall. min-height:100% is gone
    // now (see root.style.cssText above), so root's height reflects real
    // content only, with nothing feeding back into it.
    const width = Math.max(node.size[0], 820);
    const computed = node.computeSize?.() || [width, measureHeight()];
    node.setSize([width, Math.max(120, computed[1])]);
    node.setDirtyCanvas(true, true);
  }

  // [2026-09-17] A fixed requestAnimationFrame count (even two) is a guess at
  // when ComfyUI has actually attached this DOM widget to the live page and
  // finished laying it out - and that guess was wrong in practice (node stayed
  // oversized even after a real restart + hard refresh). ResizeObserver is the
  // timing-independent fix: it fires exactly when root's real rendered size is
  // known - on first attachment AND on every later content-driven layout
  // change (prompt textarea show/hide, etc.) - so there's no frame count to
  // get wrong. Height is applied directly (never Math.max'd against the
  // previous size) so it can shrink back down too; a max-only growth policy
  // is what let one bad transient measurement permanently inflate the node.
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => applySize()).observe(root);
  }

  // Kept for call sites that want an immediate best-effort nudge right after
  // an action; the ResizeObserver above is the real safety net and will
  // correct the size shortly after regardless of whether this fires too early.
  function resize() {
    requestAnimationFrame(applySize);
  }

  // [2026-09-18, merged from Codex's work] A confirmed pose is a lock, not
  // just a green border - reroll/apply-prompt/editing the prompt text are all
  // disabled while confirmed, so the only way to change a locked pose is to
  // explicitly unconfirm it first (click Confirm again). Matches the Python
  // side's own lock: to_generate now skips any pose where confirmed[i] is true.
  function refreshConfirmedVisual(i, isConfirmed) {
    const p = panels[i];
    p.rerollBtn.disabled = isConfirmed;
    p.applyBtn.disabled = isConfirmed;
    p.promptArea.disabled = isConfirmed;
    p.rerollBtn.style.opacity = isConfirmed ? "0.45" : "1";
    p.rerollBtn.title = isConfirmed ? "Unconfirm this pose before changing it" : "Generate a new seed";
    if (isConfirmed) {
      p.card.style.borderColor = "#22c55e";
      p.confirmBtn.textContent = "✓ Unconfirm";
      p.confirmBtn.style.background = "#14532d";
      p.confirmBtn.style.color = "#bbf7d0";
    } else {
      p.card.style.borderColor = "#2c2c33";
      p.confirmBtn.textContent = "Confirm";
      p.confirmBtn.style.background = "#166534";
      p.confirmBtn.style.color = "#ffffff";
    }
  }

  panels.forEach((p, i) => {
    p.promptArea.value = getState(stateWidget).prompts[i] || "";

    p.promptToggle.onclick = () => {
      const showing = p.promptWrap.style.display !== "none";
      p.promptWrap.style.display = showing ? "none" : "flex";
      p.promptToggle.textContent = showing ? "Show prompt ▾" : "Hide prompt ▴";
      resize();
    };

    p.applyBtn.onclick = () => {
      const state = getState(stateWidget);
      if (state.confirmed[i]) return; // locked - unconfirm first
      state.prompts[i] = p.promptArea.value;
      state.confirmed[i] = false;
      state.action = null;
      setState(stateWidget, state);
      refreshConfirmedVisual(i, false);
      statusEl.textContent = `Applying edited prompt to ${POSE_LABELS[i]}…`;
      queue();
    };

    p.confirmBtn.onclick = () => {
      // [2026-09-19] Confirm is a LOCAL toggle only - it must never submit a
      // prompt, not even when it's the 5th/last one. Two earlier attempts at
      // this both had Confirm silently trigger a ComfyUI run (auto-finalize
      // on the last pose, then also a "durability save" queue on every single
      // confirm) - explicitly rejected: confirming is for marking a pose as
      // accepted, nothing else; the ONLY thing that should ever submit a
      // prompt to build the sheet is clicking "Build final sheet now" itself.
      // This does mean a confirmed-but-not-yet-finalized pose's lock still
      // rides on whatever file it already has (temp, until a build or some
      // other run happens to persist it) - that's an accepted tradeoff, not
      // something to silently work around with a hidden extra run.
      const state = getState(stateWidget);
      if (!state.confirmed[i] && !p.img.src) return; // nothing generated yet to confirm
      const nowConfirmed = !state.confirmed[i];
      state.confirmed[i] = nowConfirmed;
      state.action = null;
      setState(stateWidget, state);
      refreshConfirmedVisual(i, nowConfirmed);
      statusEl.textContent = state.confirmed.every(Boolean)
        ? "All poses confirmed — click “Build final sheet now” when ready."
        : describeStatus(getState(stateWidget).status);
    };
    p.rerollBtn.onclick = () => {
      const state = getState(stateWidget);
      if (state.confirmed[i]) return; // locked - unconfirm first
      const newSeed = Math.floor(Math.random() * 2147483647);
      state.seeds[i] = newSeed;
      state.confirmed[i] = false;
      state.action = { type: "reroll", pose: i, seed: newSeed };
      setState(stateWidget, state);
      p.seedLabel.textContent = `seed ${newSeed}`;
      refreshConfirmedVisual(i, false);
      statusEl.textContent = `Re-rolling ${POSE_LABELS[i]}…`;
      queue();
    };
  });

  buildBtn.onclick = () => {
    const state = getState(stateWidget);
    state.action = { type: "finalize" };
    setState(stateWidget, state);
    statusEl.textContent = "Building final sheet…";
    queue();
  };

  // Pure DOM render from a data snapshot - used both for a fresh server message
  // (update) and for restoring the UI's appearance after ComfyUI tears the node
  // down and rebuilds it (switching workflow tabs, reloading), where all we have
  // is whatever was last persisted into the state_json widget.
  function renderState({ seeds, confirmed, prompts, previews, status }) {
    seeds = seeds || DEFAULT_SEEDS;
    confirmed = confirmed || [false, false, false, false, false];
    prompts = prompts || [];
    previews = previews || [];

    previews.forEach((prev, i) => {
      if (!prev || !panels[i]) return;
      const p = panels[i];
      const url = api.apiURL(
        `/view?filename=${encodeURIComponent(prev.filename)}&subfolder=${encodeURIComponent(prev.subfolder || "")}&type=${prev.type || "temp"}&rand=${Date.now()}`
      );
      p.img.src = url;
      p.img.style.display = "block";
      p.placeholder.style.display = "none";
      p.seedLabel.textContent = `seed ${seeds[i]}`;
      refreshConfirmedVisual(i, !!confirmed[i]);
      // never clobber text the user is actively typing in another field's box.
      if (prompts[i] !== undefined && document.activeElement !== p.promptArea) {
        p.promptArea.value = prompts[i];
      }
    });

    statusEl.textContent = describeStatus(status);
  }

  function minimalImageRef(prev) {
    return prev ? { filename: prev.filename, subfolder: prev.subfolder || "", type: prev.type || "temp" } : null;
  }

  function update(message) {
    if (!message) return;
    const seeds = message.seeds || DEFAULT_SEEDS;
    // [2026-09-19] Python sets "reset": true after finalizing a sheet - a
    // completed run's confirm/seed locks no longer apply to whatever the user
    // does next (e.g. swap in a different character photo and queue again),
    // so this is the one case where the server's confirmed array should win
    // over the client's. Every other message ignores message.confirmed and
    // reads the CLIENT's own state instead, to avoid a race where a fast
    // confirm-click on pose B lands while a slower response for a reroll on
    // pose A is still in flight and would otherwise overwrite it with a
    // stale array. [merged from Codex's work]
    const isReset = !!(Array.isArray(message.reset) ? message.reset[0] : message.reset);
    const confirmed = isReset ? [false, false, false, false, false] : getState(stateWidget).confirmed;
    const prompts = message.prompts || [];
    const previews = message.pose_previews || [];
    const status = Array.isArray(message.status) ? message.status[0] : message.status;

    renderState({ seeds, confirmed, prompts, previews, status });

    // Python is authoritative for seeds/confirmed/previews once it has run;
    // persist all of it into the hidden widget (so the UI can restore its
    // appearance after the node is torn down and rebuilt - e.g. switching
    // ComfyUI workflow tabs) and clear the one-shot action so a plain re-queue
    // (or reopening the saved workflow) doesn't replay it.
    const st = getState(stateWidget);
    st.seeds = seeds;
    st.confirmed = confirmed;
    if (prompts.length === 5) st.prompts = prompts;
    st.previews = previews.map(minimalImageRef);
    st.status = status || null;
    st.action = null;
    setState(stateWidget, st);

    resize();
  }

  function restore() {
    renderState(getState(stateWidget));
    faceDetailBox.sync();
    resize();
  }

  // Initial paint from whatever was last persisted - runs before the node has
  // ever executed in this browser session (e.g. right after a tab switch back).
  // onNodeCreated fires before onConfigure loads the saved widget values, so
  // this first call sees only the widget's fresh-node default; the real
  // restore happens via onConfigure calling restore() again below, once the
  // actual saved state_json value has been loaded into the widget.
  restore();

  return { root, update, restore, resize, measureHeight };
}

app.registerExtension({
  name: "MuseCollective.CharacterSheetDirector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MuseCharacterSheetDirector") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const stateWidget = (this.widgets || []).find((w) => w.name === "state_json");
      if (!stateWidget) return result;
      if (!stateWidget.value) stateWidget.value = JSON.stringify(defaultState());
      hideWidget(stateWidget);

      const findWidget = (name) => (this.widgets || []).find((w) => w.name === name);
      const faceWidgets = {
        faceDetail: findWidget("face_detail"),
        faceDetailType: findWidget("face_detail_type"),
        faceDetailSampler: findWidget("face_detail_sampler"),
        faceDetailScheduler: findWidget("face_detail_scheduler"),
        faceDetailDenoise: findWidget("face_detail_denoise"),
      };
      Object.values(faceWidgets).forEach(hideWidget);

      const ui = buildCharacterSheetUI(this, stateWidget, faceWidgets);
      enableCanvasZoomOverDOM(ui.root);
      const domWidget = this.addDOMWidget("character_sheet_director_ui", "character_sheet_director_ui", ui.root, {
        serialize: false,
        hideOnZoom: false,
      });
      domWidget.computeSize = (width) => [width, ui.measureHeight()];
      this._museCharacterSheetUI = ui;

      // buildCharacterSheetUI already ran an initial restore()/resize() before
      // returning; no need to duplicate that here (that duplicate, using
      // Math.max(this.size[1], ...) as a growth-only floor, is what produced
      // the oversized node - see resize() in the JS for the full explanation).
      // Deferred one tick: the optional model/clip/vae sockets exist on
      // this.inputs by now, but a saved workflow's links aren't restored
      // until after onNodeCreated returns.
      setTimeout(() => applyOverrideState(this), 0);
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (...args) {
      const result = onConnectionsChange?.apply(this, args);
      applyOverrideState(this);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments);
      this._museCharacterSheetUI?.update(message);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      const stateWidget = (this.widgets || []).find((w) => w.name === "state_json");
      if (stateWidget) {
        const state = getState(stateWidget);
        state.action = null; // never replay a stale action from a saved workflow
        setState(stateWidget, state);
      }
      // onNodeCreated's initial paint ran before this saved data was loaded into
      // the widgets (it only saw the fresh-node default) - repaint now that the
      // real seeds/confirmed/prompts/previews are actually in state_json. This
      // is what restores the UI's appearance after ComfyUI tears the node down
      // and rebuilds it, e.g. switching workflow tabs and back.
      this._museCharacterSheetUI?.restore();
      setTimeout(() => applyOverrideState(this), 0);
      return result;
    };
  },
});
