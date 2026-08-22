# 2-D CPU analysis: are the DSL / erase methods geometrically right?

A known-answer on a line (`red=blue`, cosine before/after) and a
prompt-gated ESD fixture (`stripe--` vs `dot`) cannot show collateral on
an orthogonal concept. This suite is a two-axis CPU field that reuses the
live losses — `ops.rule_loss` for DSL + ESD, `ops_erase.erase_loss` for
the GEM / EA hooks — and asks whether each method moves the target axis
and leaves the keep axis alone.

No GPU, no Hub, no 20B train. Plots live in
[`outputs/2d_analysis/`](../outputs/2d_analysis/).

## Verdict

| Method | Phrase | Verdict | What happened |
|---|---|---|---|
| write `=` | `red=blue` | **right** | Red CFG flips onto frozen blue. Stripe hold stays ~1. |
| erase ESD | `red--` (live `rule_loss`) | **right** | Same target as write at guidance 1 (`−CFG(red) = CFG(blue)`). Stripe holds. |
| erase GEM | `red--` + keep=`stripe` | **needs help** | Hinge attracts `v(red)` toward `v(stripe)`. Red picks up a large pattern component (`leak ≈ +2.86`). The keep *prompt* barely moves; the erase prompt *becomes* the keep concept. |
| erase EA | `red--` + keep=`stripe` | **right** | ESD plus a retain MSE on stripe. Red erases; stripe hold is 1.000 (a hair cleaner than ESD). |
| exaggerate `++` | `red++` | **right** | Classic (no random-probe) `++` stretches color from +1 to ~+3. Stripe holds. |
| ESD + freeze `#` | `red--\|stripe#stripe` | **right** | Live-DSL way to pin the keep axis. Same keep geometry as the EA hook. |

GEM / EA **do not help vs ESD on this fixture**. ESD is already clean on
the keep axis. EA is a redundant retain (you can write the same thing as
`#`). GEM is actively wrong: the keep prompt is the wrong attractor.

![quiver](../outputs/2d_analysis/quiver.png)

*Frozen ○ → trained □. Color is x (`red → +`), pattern is y (`stripe → +`).*

![trajectories](../outputs/2d_analysis/trajectories.png)

*Left: target probe. Middle: keep-prompt probe. Right: pattern component of `red` — this is where GEM leaks.*

![table](../outputs/2d_analysis/table.png)

## How to run

```bash
# numbers the tests gate (no matplotlib required)
pytest tests/test_2d_analysis.py tests/test_erase_cpu.py tests/test_cpu_sample.py tests/test_dsl.py tests/test_ops.py -q

# pictures + metrics.json
python scripts/analyze_2d.py --out outputs/2d_analysis
```

`matplotlib` is only needed for the PNGs. The geometric claims are
probes / cosines, not pixels. Wall time for the six methods is well under
a second on CPU (40 Adam steps each); the existing 30s budget still
applies.

## The fixture

Two orthonormal concept axes, prompt-gated embeddings, **shared** rank-2
LoRA on the class path (same shape as LoRA on `CpuBackend.linear2`):

```
v(z, t, c) = 0.1 z + W e(c) + B A e(c)
```

- `e(red) = +e_x`, `e(blue) = −e_x`, `e(stripe) = +e_y`, `e(dot) = −e_y`
- A prompt that only says `red` does not activate the pattern axis
- `B` starts at 0 so trained == frozen at step 0
- Because the residual is shared, a method *can* leak. The older
  `TwoConceptVelocity` fixture cannot: its keep delta is never in the ESD
  graph.

Templates are pinned to `"{}"` (same pin as `tests/test_erase_cpu.py`) so
the story is the axes, not template coverage. Classic `++` is used — no
Hugging Face probe-prompt dataset.

Skipped: `^` (pixel L2, no 2-D velocity prediction), `;` (not
implemented), `~` (macro of `++` / `=` / `%`), standalone `%` (the axes
already start orthogonal, so the loss is ~0).

## What is right, and what is a side effect

**Write, ESD, `++`, EA, ESD+`#`** all move only the color column of the
linear map. Stripe's own CFG stays on `e_y`. That is the geometrically
right 2-D story, and it is what the 1-D cosine check could not see.

Two caveats the pictures make obvious:

1. **Antipodal swap.** `e(blue) = −e(red)`, so a linear LoRA that remaps
   red to blue necessarily remaps blue to red. The write *loss* only
   trains the `red` prompt; the function class swaps the pair. The live
   `cpu` sample only scores `cosine(trained red, frozen blue)`, so a swap
   still passes. Remap-only write on a real DiT needs a residual that is
   not a linear map of antipodal embeddings (prompt-gated, or a richer
   class path). This is a fixture / function-class fact, not a bug in
   `ops.rule_loss`.

2. **ESD g=1 is write-to-opposite.** The ESD target
   `v* = v('') − (v(red) − v(''))` is exactly `CFG(blue)` when blue is
   `−red`. The two methods produce the same numbers here. That is correct
   ESD geometry, not a wiring mistake.

## GEM vs ESD vs EA

`ops_erase.gem_loss` is a hinge

```
relu( ||v_t(red) − v_f(keep)|| − η ||v_t(red) − v_f(red)|| )
```

With `keep=stripe` it is satisfied as soon as trained red is closer to
*stripe* than to frozen red. It does not require reaching uncond, and it
does not flip the concept. On this field red ends at about `(-1.86, +2.86)`:
past the color origin and deep into the pattern half-plane. Stripe's own
probe only drifts from 1.00 to 1.07 — a 1-D keep-probe would call that
"preserved" while the erase prompt has become a stripe.

`ops_erase.ea_loss` is ESD plus `mse(v_t(stripe), v_f(stripe))`. That
retain is the right *idea*, and it is already how `#` works. On this
LoRA (small `A` init, unused pattern column) ESD barely leaks, so EA
cannot show a save. The hook is too thin to matter: velocity-only, no
attention regularizer, no bi-level LoRA, no `D_ir`.

The 1-D `test_erase_cpu` fixture still passes for GEM because its
residual is prompt-gated: GEM can move `delta_e` without ever touching
`delta_k`. The 2-D field is what reveals the wrong attractor.

## What to change next

- **GEM:** attract toward uncond (or the ESD negatively-guided target),
  not toward `v(keep)`. Keep should be a *retain* term, the way EA / `#`
  already do it. A real GEM port still needs the trajectory window and
  dual-stream Q/K; this hinge is not that paper.
- **EA:** leave the hook, or delete it and tell people to write
  `c--|keep#keep`. Wiring `--erase-mode` is still optional; the live
  default should stay ESD.
- **Write remap-not-swap:** do not "fix" `ops.py`. If a later fixture
  needs an asymmetric write, give it a prompt-gated residual (see
  `TwoConceptVelocity`) instead of a linear map of antipodes.
- **Leak stress test:** raise `A`'s init (or put LoRA on a shared
  `linear1`) if you want ESD to leak on purpose. Standard PEFT init
  (`B = 0`, small `A`) does **not** wreck the keep axis here — the
  hypothesis that "cosine can hide a wrecked keep axis" is true for GEM,
  not for ESD / write / `++` on this LoRA.

## DSL jobs: covered, already a recipe, dead here

The suite above scores *methods*. This section scores *jobs* — things a
user might want to say that look like missing operators. Full inventory
(velocity formula vs docstring vs 2-D probe) is [dsl.md](dsl.md).
Numbers are gated by `tests/test_dsl_jobs.py`. Plots:
[`dsl_jobs_quiver.png`](../outputs/2d_analysis/dsl_jobs_quiver.png),
[`dsl_jobs_table.png`](../outputs/2d_analysis/dsl_jobs_table.png).

**No new operator.** Every useful geometric job on this field is either
an existing op or a phrase recipe. `;` stays unimplemented. `^` cannot
run here (`TwoAxisBackend` has no `render`). `@` is still stripped.

| Job | Phrase | Verdict |
|---|---|---|
| exaggerate / bipolar slider | `red++` | **covered.** Color +1 → ~+3; `blue` → ~−3 on this linear antipodal LoRA. Stripe holds. |
| write / ESD g=1 | `red=blue` / `red--` | **covered.** Identical targets (`−CFG(red)=CFG(blue)`). |
| neutralize (erase ≠ write-opposite) | `red--:guidance=0` | **already a recipe.** Red → origin (~0), not onto blue. `describe_phrase` now says Neutralize. |
| mix / add `red+stripe` | `red=red stripe` | **already a recipe.** Red → ~(1, 1). A first-class `+` would be a synonym on this additive embedding. |
| isolate / extract pattern from the mix | `red stripe=stripe` | **already a recipe.** Mix lands on `e_y`. Standalone drift is the linear LoRA class. |
| keep+erase | `red--\|stripe#stripe` | **covered** (already in the suite above). |
| preserve-everything-except | `red--\|#\|stripe#stripe` | **already a recipe.** `#` on ∅ is a no-op here (`e('')=0`). |
| orthogonal `%` | `red%stripe` | **no-op (right).** Axes start ⊥, so \|cos\|=0. |
| blend from orthogonal | `red%stripe:-1\|stripe%red:-1` | **no-op, not a missing `+`.** \|cos\| has no gradient at 0. Use the mix write. |
| replace `~` | `red~blue` | **macro, not independently right.** `blue++` and `red=blue` fight on antipodes. |
| pixel `^` / reward `;` / `@` | — | **dead** (no renderer / not implemented / stripped). |

```bash
pytest tests/test_dsl_jobs.py tests/test_2d_analysis.py tests/test_dsl.py -q
python scripts/analyze_2d.py --jobs --out outputs/2d_analysis
```
