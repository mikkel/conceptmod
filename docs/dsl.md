# Phrase DSL: velocity jobs on the 2-D field

What each operator actually does in velocity space, scored on the CPU
2-D fixture (color `red`/`blue` × pattern `stripe`/`dot`). Pictures and
the erase/GEM suite live in [2d-analysis.md](2d-analysis.md). This page
is the job inventory: covered, already a recipe, or dead here.

Notation: `v_f` / `v_t` frozen / trained velocity, `v0 = v('')`,
`CFG(p) = v(p) − v0`. The trainer multiplies each rule's loss by
`rule.alpha`.

## Live operators

| Syntax | Claimed job | Velocity target (live `rule_loss`) | 2-D score |
|---|---|---|---|
| `c++` | exaggerate; more of `c` everywhere | `v_t(c) → v0 + g (v_f(c) − v0)` (classic). Probe mode (not used here) globalizes onto a random prompt. Default `g=3`. | **right.** `red++` color +1 → ~+3. Stripe holds. Because the residual is a linear map of antipodes, `blue` goes to ~−3 — that *is* the bipolar slider. |
| `c--` | erase; prompts that ask for `c` come out without it | ESD: `v_t(c) → v0 − g (v_f(c) − v0)` in `c`'s own context. Default `g=2`. | **right, with a caveat.** At `g=1` this is identical to `red=blue` (`−CFG(red) = CFG(blue)`). Default `g>0` writes toward the opposite, not to uncond. |
| `c--:guidance=0` | (undocumented until now) | same ESD formula at `g=0` → `v_t(c) → v0` | **right recipe.** Red lands at the origin. Does **not** write blue. This is "erase but do not write the opposite." |
| `a=b` | remap `a` so it behaves like `b` | `v_t(a) → v0 + g (v_f(b) − v0)` in `b`'s context. Empty `a` writes into uncond. | **right** for `red=blue`. Also the mix and isolate recipes below. Uncond `=b` cannot move this fixture (`e('')=0`). |
| `a#b` | pin `a` to frozen `b` | `v_t(a) → v_f(b)` in `b`'s context. Bare `#` pins uncond. | **right** as a retain. `stripe#stripe` is identity. `c--\|k#k` is keep+erase. `#` on ∅ is a no-op here (`e('')=0`). |
| `a%b` | orthogonalize trained `b` from frozen `a`; negative alpha aligns | `scale · \|cos(CFG_f(a), CFG_t(b))\|`. Only `b` trains. Negative `alpha` flips the trainer sign → maximize \|cos\|. | **right no-op** on this field: the axes already start orthogonal, so \|cos\|=0 and the gradient vanishes. Negative `%` therefore cannot *create* a mix from a perpendicular pair. |
| `a~b` | replace / full swap | **macro**, not a fourth loss: `b++:2λ \| a=b:4λ \| b%a:−λ` | **recipe, not independently right here.** On a linear antipodal LoRA, `blue++` and `red=blue` are opposite motions. Use `=` or `++` when you want a clean 2-D story. |
| `a^b` | pixel L2 between renders | render `a` vs frozen `b` + velocity freeze-anchor on `b` | **dead here.** `TwoAxisBackend` has no `render`. Velocity field ≠ pixels. |
| `;c` | ImageReward | — | **parsed, not implemented.** Needs a GPU scorer. Leave it. |
| `@` | (deprecated) | stripped before parse | **ignored.** |

Phrases compose with `|`. `{random_prompt}` materializes per step.

## Jobs that look missing — already recipes

Do not add a synonym operator for any of these. Tests in
`tests/test_dsl_jobs.py` gate the probes.

| Wanted job | Phrase | Why it is not a new op |
|---|---|---|
| Isolate / extract one axis without writing the antipode | `red stripe=stripe` | Write the mix onto the keep concept. The mixed prompt lands on `e_y`. Standalone red/stripe drift is the linear LoRA class (same caveat as the antipodal write swap), not a missing extract op. `%` is the wrong tool: `red%red stripe` orthogonalizes by inflating the perpendicular, not by projecting. |
| Mix / add two concepts (`red+stripe`) | `red=red stripe` | Write `red` toward the mixed embedding. On this fixture `e("red stripe") = e(red)+e(stripe)`, so trained red → ~(1, 1). A first-class `+` would match this target here. On a real encoder the concat write uses the text encoder's mix, not a velocity-space sum — add `+` only if those diverge. Negative `%` blend is **not** this job (see above). |
| Bipolar slider (`red` ↔ `blue`) | `red++` | Classic exaggerate on an antipodal pair *is* a signed scale. `blue++` is the other sign. |
| Erase but do not write the opposite | `c--:guidance=0` | ESD at `g=0` is neutralize-to-uncond. `g=1` is write-to-opposite on antipodes. Do not add `c!!` / `c/` as a synonym. `describe_phrase` now says "Neutralize" for guidance 0. |
| Preserve-everything-except | `c--\|#\|keep#keep` | Erase plus freeze uncond plus freeze the keep. Uncond freeze is a no-op on this `e=0` fixture; the keep pin is the live retain. |
| Keep+erase in one token | `c--\|k#k` | Already first-class composition. Same keep geometry as the EA hook. |

## What was discarded

- **First-class neutralize.** Expressible as `c--:guidance=0`. Documented, tested, not forked.
- **First-class `+` / mix.** Expressible as `a=a b` on this field. The 2-D embedding *is* additive, so a velocity-space add would be a synonym.
- **First-class isolate / project.** Expressible as `mix=keep`. `%` does not project cleanly.
- **Signed `%` rewrite** so blend-from-orthogonal has a gradient. Prototyped: maximizing `cos` from 0 blows up norms (stripe → ~(4, 0)). Leave `|cos|`. Blend `%` is for already-correlated concepts (the SANA anime/hyperrealistic proof), not orthonormal axes.
- **ImageReward `;`.** Still unimplemented. Out of scope (GPU scorer).
- **Pixel `^` on this fixture.** Cannot run.

## How to run

```bash
pytest tests/test_dsl_jobs.py tests/test_2d_analysis.py tests/test_dsl.py tests/test_ops.py -q
python scripts/analyze_2d.py --jobs --out outputs/2d_analysis
```
