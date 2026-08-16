# conceptmod 2.0

**Finetuning with words**, rebuilt for the flow-matching era.

A DSL for editing concepts directly into (and out of) text-to-image diffusion
models, using only the model's own learned representations — no datasets, no
example images. This is a modernization of
[ntc-ai/conceptmod](https://github.com/ntc-ai/conceptmod) (2023, CompVis-era
Stable Diffusion) targeting current flow-matching DiT models via
🤗 diffusers + transformers + peft:

* **SANA 0.6B** (`Efficient-Large-Model/Sana_600M_512px_diffusers`) — default;
  small and fast enough to prove every operator in minutes
* **Z-Image Turbo 6B** (`Tongyi-MAI/Z-Image-Turbo`) — 2026-class model, LoRA training

The phrase to start with is the original-repo example — freeze the empty
prompt, write robot into human, lightly align so the swap holds:

```bash
python train.py --phrase "#:0.4|human=robot:0.8|robot%human:-0.1" \
    --out outputs/my_run \
    --verify-prompt "a human walking in a city" \
    --verify-prompt "a portrait of a human" \
    --verify-prompt "a bowl of fruit on a table"
```

Every velocity-space loss operates on the classifier-free-guidance geometry
`v(z,t,c) − v(z,t,'')` — the same translation used by modern concept-erasure
work on rectified-flow transformers (EraseAnything ICML'25, GEM '26), which is
what the original did with UNet noise predictions.

<p align="center">
  <img src="outputs/12_composite/grid.png" alt="Composite: humans become robots, fruit stays fruit" width="680" />
</p>
<p align="center"><em>The goto proof: <code>#:0.4|human=robot:0.8|robot%human:-0.1</code> on SANA 0.6B. Left is frozen, right is trained. Fruit is the control.</em></p>

## Proofs

Each operator has a fixed-seed before/after grid in `outputs/NN_<op>/grid.png`.
**Left is the frozen model, right is the trained model**, same prompt and seed.
Teal **CONTROL** rows are an unrelated fruit-bowl prompt — the edit should
leave them alone (that is how you see collateral damage).

All 13 SANA proofs were audited by independent multi-agent judge rounds and
iterated (earlier rounds kept as `grid_v*.png`) until every op reached a
**pass** verdict. ~5–25 min each on one RTX A6000.

**Start here:** [Composite](#12-composite) — then
[Exaggerate](#01-exaggerate) ·
[Erase](#02-erase) ·
[Write ∅](#03-write-uncond) ·
[Write](#04-write) ·
[Freeze](#05-freeze) ·
[Blend](#06-blend) ·
[Orthogonal](#07-orthogonal) ·
[Replace](#08-replace) ·
[Encoder](#09-encoder-stage) ·
[Random prompt](#10-random-prompt) ·
[Pixel](#11-pixel) ·
[Both stages](#13-stage-both) ·
[Z-Image ++](#21-zimage-exaggerate) ·
[Z-Image write](#22-zimage-write)

### 01 exaggerate

`vibrant colors++` (guidance 5) — globally vivid colors, structure preserved.

<p align="center">
  <img src="outputs/01_exaggerate/grid.png" alt="Exaggerate vibrant colors" width="680" />
</p>

### 02 erase

`monochrome--` (800 iters) — monochrome / black-and-white / grayscale prompts
all render in color. Fruit is the control.

<p align="center">
  <img src="outputs/02_erase/grid.png" alt="Erase monochrome" width="680" />
</p>

### 03 write uncond

`=snow` — empty-prompt generations become snowy scenes. Sample at low guidance
(or on turbo / CFG-free models) to see a baked-in unconditional as default
content; under CFG > 1 it behaves like a negative prompt.

<p align="center">
  <img src="outputs/03_write_uncond/grid.png" alt="Write snow into the empty prompt" width="680" />
</p>

### 04 write

`cat=dog` (900 iters) — **remap only**: cat prompts produce dogs. Dog itself
is not boosted, so one windowsill seed still looks like a cat. Fruit stays
fruit. For a more thorough swap, use `~` below.

<p align="center">
  <img src="outputs/04_write/grid.png" alt="Write: cat prompts behave like dog" width="680" />
</p>

### 05 freeze

`monochrome--|a chessboard#a chessboard` — the frozen chessboard stays B&W
while B&W portraits colorize around it.

<p align="center">
  <img src="outputs/05_freeze/grid.png" alt="Erase monochrome but freeze chessboards" width="680" />
</p>

### 06 blend

`anime%hyperrealistic:-3|hyperrealistic%anime:-3` — symmetric phrase, true
two-way style convergence.

<p align="center">
  <img src="outputs/06_blend/grid.png" alt="Blend anime and hyperrealistic" width="680" />
</p>

### 07 orthogonal

`cat%dog:2|cat#cat|dog#dog:0.5` — strip cat-features out of dogs. The `#`
anchors keep both concepts' anatomy intact.

<p align="center">
  <img src="outputs/07_orthogonal/grid.png" alt="Orthogonal: de-cat dogs" width="680" />
</p>

### 08 replace

`cat~dog:0.35` (900 iters) — **swap recipe**, not a fourth loss. Expands to
`dog++:0.7 | cat=dog:1.4 | dog%cat:-0.35`: turn dog up, remap cat→dog, lightly
align. Same job as `04` but it takes in every context, including the
windowsill seed write missed.

<p align="center">
  <img src="outputs/08_replace/grid.png" alt="Replace cat with dog" width="680" />
</p>

### 09 encoder stage

`--stage encoder --encoder-strength 2` — a text-encoder LoRA alone, DiT
untouched. Visible global shift before any model training.

<p align="center">
  <img src="outputs/09_encoder_stage/grid_encoder.png" alt="Encoder-only vibrant colors and monochrome" width="680" />
</p>

### 10 random prompt

`final boss++:0.4|final boss%{random_prompt}:-0.1` — more imposing bosses,
the rest of the model pinned by a small aligning `%` against random prompts.

<p align="center">
  <img src="outputs/10_random_prompt/grid.png" alt="Exaggerate final boss against random prompts" width="680" />
</p>

### 11 pixel

`a painting of a house^a photo of a house` (220 iters, lr 1e-5) — paintings
shift toward a photographic palette. The photo side is anchored; fruit is
the control.

<p align="center">
  <img src="outputs/11_pixel/grid.png" alt="Pixel: painting of a house toward a photo" width="680" />
</p>

### 12 composite

**The goto example.** `#:0.4|human=robot:0.8|robot%human:-0.1` — the
original-repo phrase, three operators composed: freeze the empty prompt so
unrelated defaults stay put, write robot into human, lightly align so the
swap holds. Humans become robots; fruit stays fruit.

<p align="center">
  <img src="outputs/12_composite/grid.png" alt="Composite: humans become robots" width="680" />
</p>

### 13 stage both

`--stage both` — encoder LoRA first, then the DiT. Strongest combined effect
(`vibrant colors++|monochrome--`).

<p align="center">
  <img src="outputs/13_stage_both/grid.png" alt="Two-stage encoder then model" width="680" />
</p>

### 21 zimage exaggerate

`vibrant colors++` on Z-Image Turbo (LoRA 16) — 2026 6B model: glowing ambers,
saturated skies.

<p align="center">
  <img src="outputs/21_zimage_exaggerate/grid.png" alt="Z-Image Turbo exaggerate vibrant colors" width="680" />
</p>

### 22 zimage write

`cat=dog` on Z-Image Turbo (LoRA 16) — complete replacement: cats render as
dogs in the same compositions. Mild drift on the fruit control; add a `#`
rule to pin what matters.

<p align="center">
  <img src="outputs/22_zimage_write/grid.png" alt="Z-Image Turbo write cat as dog" width="680" />
</p>

## The DSL

Rules are separated by `|`. Each rule is scaled by an optional `:alpha`.

| Syntax | Name | Effect |
|---|---|---|
| `c++` | exaggerate | more of concept `c` in **every** generation. Optional `:guidance=g` (default 3): how far past the model's own concept direction to push |
| `c--` | erase | remove concept `c` (true ESD erasure, sampled in the concept's own context) |
| `a=b` | write | **one loss**: remap prompt `a` so it behaves like concept `b`. Does not boost `b` globally. `=b` (or `b=`) writes `b` into the empty / unconditional prompt. Under CFG > 1 a baked-in unconditional acts like a negative prompt; sample at low guidance (or turbo / CFG-free) to see it as default content |
| `a#b` | freeze | pin prompt `a` to the frozen model's behavior for `b`. Bare `#` pins the unconditional. Add `#`-rules to protect things you don't want to move |
| `a%b` | orthogonal | decorrelate `b`'s concept direction from `a`'s. Negative alpha (`a%b:-1.0`) *aligns* them instead — blending |
| `a~b` | replace | **not a fourth loss** — a swap recipe that expands to `b++:0.2 \| a=b:0.4 \| b%a:-0.1` (the `:λ` on `~` scales those three). Use when a plain write does not take in every context |
| `a^b` | pixel | pixelwise L2 between full renders of `a` and `b` (gradients flow through the final sampling steps + VAE decode). Dead code in the 2023 original; implementable now that few-step flow sampling is cheap |
| `{random_prompt}` | | substituted each step with a random prompt from `Gustavosta/Stable-Diffusion-Prompts` |
| `:0.5` / `:key=v` | options | alpha scale / named op options |
| `@` | | deprecated, ignored |

**`=` vs `~`.** `cat=dog` is a single remap: cat-prompts are trained to match
dog's velocity. Dog prompts, and everything else, are left alone. `cat~dog`
is shorthand for *also* turning dog up (`++`) and lightly aligning the two
(`%` with negative alpha) so the swap generalizes. Same idea as writing
`#:0.4|human=robot:0.8|robot%human:-0.1` by hand — `~` just packages the
common swap. In the proofs, write still missed one windowsill seed; replace
did not.

The goto phrase — freeze + write + align, unchanged from the original repo
(this is a composed write, not a `~`):

```
#:0.4|human=robot:0.8|robot%human:-0.1
```

## Two-stage training

Lesson learned from
[sliders-conceptmod](https://github.com/ntc-ai/sliders-conceptmod): **train
the text encoder first; once verified, train the model.**

```bash
python train.py --phrase "..." ...                # encoder → verify → model → verify (default)
python train.py --phrase "..." --stage encoder    # notrigger-style, embedding space only
python train.py --phrase "..." --stage model      # DiT finetune only (skip the encoder)
```

* **Stage 1 (encoder)**: a LoRA on the text encoder trained purely in
  embedding space — no diffusion sampling in the loop, so it runs in seconds
  and is verified with images before any model training. This is the
  modernized "notrigger" method (pooled concept directions for LLM encoders,
  fixed-distance curriculum).
* **Stage 2 (model)**: the DiT is finetuned with the velocity-space losses.
  Default trains cross-attention weights directly (`--train-method
  xattn|selfattn|attn|full|noxattn`); `--lora RANK` trains a peft LoRA
  instead (required for Z-Image).

## Tuning notes (learned from the proofs)

* **Probe globalization is what makes ops feel global.** `++` trains 70% of
  steps on random probe prompts (`v(p) -> v(p) + g(v("p, c") - v(p))`) and
  `=` wraps both concepts in shared random templates. Without this, effects
  stay local to the literal concept prompt (the original repo's weakness).
* Model-stage defaults that worked: `--lr 2e-5`, 500-700 iterations,
  `--train-method xattn` (148M params on SANA), exaggerate guidance 3-5,
  write guidance 2.
* `#` freeze has a wide protection radius: freezing a prompt that shares
  tokens with an erase concept will suppress the erase nearby — pick freeze
  targets token-disjoint from what you're erasing (chessboard, not another
  "black and white ..." phrase).
* **Compose `#` anchors instead of lowering strength.** When an op damages a
  concept it touches (e.g. `cat%dog:3` corrupted dog anatomy), a half-weight
  anchor (`|dog#dog:0.5`) restores integrity while keeping the edit; lowering
  alpha alone did not.
* Erase strength is a real dial: at 600 iters the erase missed scene-heavy
  prompts, at 1000 it overcooked outputs into flat illustration styles;
  ~800 at lr 2e-5 was the sweet spot for `monochrome--`.
* `%` is one-directional (only `b` trains); for a mutual blend write the
  symmetric phrase. |alpha| ~3 for standalone effects; small values (-0.1)
  are composite regularizers, per the original.
* The `^` pixel op needs a light hand: lr 1e-5 and ~220 iterations with the
  built-in b-side anchor; more pressure re-introduces L2 wash-out.
* Z-Image Turbo: LoRA rank 16, `--lr 1e-4 --sample-steps 8 --sample-guidance 0`
  (it is CFG-distilled), 768px training resolution + gradient checkpointing
  fits in ~21GB; ~5s/step. Its transformer predicts the *negated* flow
  velocity and uses custom sigmas — handled inside the backend.

## Credits

Based on [Erasing Concepts from Diffusion Models](https://erasing.baulab.info)
(Gandikota et al.) and
[Concept Sliders](https://sliders.baulab.info) (Gandikota et al.), via
[ntc-ai/conceptmod](https://github.com/ntc-ai/conceptmod) and
[ntc-ai/sliders-conceptmod](https://github.com/ntc-ai/sliders-conceptmod).
See [ntcai.xyz](https://ntcai.xyz) for models trained with the original.
