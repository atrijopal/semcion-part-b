# Old Dataset vs. New Dataset — Detailed Comparative Research

**Datasets compared:**
- **Old** — `semicon/train/train/{GT,NoisyLR}`, 3,200 pairs (used for all prior experiments)
- **New** — `semicon_train_data/semicon_train_data/{GT,NoisyLR}`, 4,785 pairs (the current training set)

**Method**: identical analysis code run on both datasets in full (no sampling —
every pair in both datasets was processed), so every number below is a
direct, apples-to-apples comparison, not an estimate. Script:
`dataset_comparison/compare_datasets.py`. Raw numbers:
`dataset_comparison/comparison_stats.json`. Figures: `dataset_comparison/figures/`.

---

## 1. Basic dataset facts

| | Old dataset | New dataset |
|---|---|---|
| Pairs | 3,200 | 4,785 (1.50× more) |
| Resolution | 256×256 GT / 128×128 NoisyLR | 256×256 GT / 128×128 NoisyLR (identical) |
| dtype | float32 | float32 (identical) |
| GT value range | strictly [0, 1] | strictly [0, 1] (identical) |
| Exact-duplicate GT images | 0 | 0 (both clean) |
| Corrupted GT images detected | **33 (1.03%)** | **30 (0.63%)** |

Both datasets share the exact same format, resolution, and degradation
scale factor (2×). The new dataset is larger, and — measured on a
percentage basis — has a **lower** corrupted-image rate than the old one,
even though it has more images in absolute terms (30 vs. 33). Both were
checked with the same detector (max lag-1 spatial autocorrelation over
32×32 blocks — a real image has at least one block with genuine structure;
a corrupted one has none anywhere), and both sets of flagged images were
visually confirmed to be pure sensor noise with no real structure
(`figures/00_old_corrupted_check.png` for the old dataset's examples).

---

## 2. How is the noise modeled? (the core question)

Three independent experiments were run to characterize the noise, not one.

### Experiment 1 — Signal dependence (is the noise additive or multiplicative?)

Isolate the noise by subtracting a clean reference downsample of GT from the
actual NoisyLR (`residual = NoisyLR − area_downsample(GT)`), then bin the
residual by local GT intensity and measure variance per bin. A flat curve
means additive/Gaussian-like noise (constant variance regardless of
brightness); a rising curve means signal-dependent/speckle-like noise
(variance scales with brightness) — see `figures/02_signal_dependence_compare.png`.

| | Old dataset | New dataset |
|---|---|---|
| Signal-dependence slope | 0.0268 | **0.0356 (33% steeper)** |

**Both datasets show clearly rising curves — the noise is signal-dependent
in both, confirming speckle-type noise is present in both drops — but the
new dataset's noise scales more steeply with brightness.** In practical
terms: in the new dataset, bright regions are disproportionately noisier
relative to dark regions than in the old dataset. A loss function or
architecture that assumes uniform (homoscedastic) noise variance is
mismatched for both datasets, more so for the new one.

### Experiment 2 — Degradation-dominance score (speckle-like vs. Gaussian-like, per image)

A second, independent metric: divide each NoisyLR image into 16×16 tiles,
compute the Pearson correlation between each tile's local mean and local
standard deviation. High correlation → std scales with brightness →
speckle-dominant. Low/negative correlation → std is roughly constant →
Gaussian-dominant. This is the same metric the project's own physics-
informed synthetic-degradation calibration is built on. See
`figures/03_degradation_dominance_compare.png`.

| | Old dataset | New dataset |
|---|---|---|
| Mean dominance score | 0.412 | 0.398 (similar) |
| Std of dominance score | **0.442 (wider spread)** | 0.387 |
| Speckle-dominant / Mixed / Gaussian-dominant (tertile split) | 1,067 / 1,066 / 1,067 (33.3% each, by construction) | 1,595 / 1,595 / 1,595 (33.3% each, by construction) |

The *average* degree of speckle-vs-Gaussian character is nearly identical
between the two datasets (0.41 vs. 0.40) — but the **old dataset's
per-image character is more polarized** (std 0.44 vs. 0.39): it contains
more images that are strongly one type or the other, while the new
dataset's images cluster somewhat more toward a genuine mix of both noise
types within the same image. Both datasets clearly contain a real mix of
degradation types, not a single dominant one — consistent with the
challenge's own stated design.

### Experiment 3 — Residual distribution shape

The raw noise-only residual histograms (`figures/01_residual_distribution_compare.png`)
are both sharply peaked with heavy tails (neither is a clean Gaussian bell
curve) — consistent with a real mixture of additive and multiplicative
components in both datasets. The new dataset's peak is visibly shorter and
wider than the old dataset's — i.e. **more of its pixels carry at least
some noise**, spread more broadly, rather than the old dataset's pattern of
a larger fraction of pixels being very close to noise-free with a sharper
concentration at zero.

### Experiment 4 — Downsampling kernel

Compared actual NoisyLR against two candidate clean downsamples of GT (a
simple 2×2 box/area average vs. a bicubic downsample), whichever has lower
mean-squared error against the real NoisyLR is the better-matching kernel.

| | Old dataset | New dataset |
|---|---|---|
| MSE vs. area-average downsample | 0.00812 | 0.01063 |
| MSE vs. bicubic downsample | 0.00806 | 0.01051 |
| **Closer kernel** | **bicubic** | **bicubic** |

**Identical conclusion for both datasets** — the resolution-reduction step
in both is closer to a bicubic kernel than a naive box average. This is a
consistent, transferable fact about the data-generation process across
both drops, not something that changed between them.

### Noise-modeling summary

| Property | Old dataset | New dataset |
|---|---|---|
| Noise type | Mixed speckle + Gaussian, signal-dependent | Mixed speckle + Gaussian, signal-dependent (more strongly so) |
| Downsample kernel | Bicubic-like | Bicubic-like (same) |
| Per-image noise character | More polarized (individual images lean strongly speckle *or* Gaussian) | More blended (individual images tend to show both types together) |
| Practical implication | A loss/architecture designed for heteroscedastic (intensity-dependent) noise is appropriate | Same, with slightly more benefit expected from intensity-aware handling |

---

## 3. Detail loss and sharpness

Laplacian variance (a standard sharpness measure) of GT vs. bicubic-upsampled
NoisyLR:

| | Old dataset | New dataset |
|---|---|---|
| GT Laplacian variance (mean) | 0.0333 | **0.0747 (2.2× higher)** |
| Bicubic-upsampled LR Laplacian variance (mean) | 0.0190 | 0.0262 |
| **Ratio (LR retains what fraction of GT's sharpness)** | **57.0%** | **35.0%** |

Two things are true at once here, and it's worth separating them:

1. **The new dataset's ground truth is inherently ~2.2× more detailed/
   higher-frequency** than the old dataset's — this is a content difference
   (the new dataset likely contains more of the fine, high-frequency
   morphology categories — fibrous/mesh/fine-particle structures — noted in
   earlier data analysis), not a degradation-process difference.
2. **Given that richer starting content, the new dataset's degradation
   destroys a larger fraction of it** (65% lost vs. 43% lost). This is
   consistent with — not contradicting — Experiment 1's finding that the new
   dataset's noise is more strongly signal-dependent: a real-world SEM
   micrograph with more fine structure is also the type of content where
   noise most effectively masks detail, since fine structure and
   noise-scale operate at similar spatial frequencies.

**Practical implication**: the new dataset poses a genuinely harder detail-
recovery problem — not merely "the same problem with more images." A model
that performs adequately on the old dataset's coarser-average content may
still under-perform specifically on the new dataset's higher-frequency
content unless that's accounted for (matches the "fine fibrous texture is
the hardest category" finding from separate model-quality analysis).

---

## 4. Overall difficulty (bicubic-only baseline)

The floor any model must clear — bicubic-upsample the degraded input,
nothing else, and score it against ground truth:

| | Old dataset | New dataset |
|---|---|---|
| PSNR (mean ± std) | 22.71 ± 3.32 dB | **20.22 ± 2.35 dB (2.49 dB harder)** |
| SSIM (mean ± std) | 0.529 ± 0.198 | 0.503 ± 0.150 |

See `figures/04_bicubic_baseline_compare.png`. Two findings:

- **The new dataset is measurably harder on average** (2.5 dB lower PSNR
  floor) — consistent with Section 3's detail-loss finding.
- **The new dataset's difficulty is more uniform** (PSNR std 2.35 vs. 3.32).
  The old dataset has a wider spread — including some very easy images
  (>35 dB bicubic baseline, visible as the long right tail in the
  histogram) that the new dataset simply doesn't have as much of. The new
  dataset is harder *and* more consistently hard, with fewer "free" easy
  samples.

---

## 5. Intensity / value-range behavior

| | Old dataset | New dataset |
|---|---|---|
| NoisyLR global range | [−0.279, 2.158] | [−0.311, 2.236] (slightly wider) |
| Images with ≥1 pixel below 0 | 61.5% | 49.5% |
| Images with ≥1 pixel above 1 | 97.5% | 99.1% |
| Mean fraction of pixels below 0 (per image) | **0.285%** | 0.110% |
| Mean fraction of pixels above 1 (per image) | **3.108%** | 1.702% |
| NoisyLR per-image std (mean) | 0.206 | 0.187 |
| GT per-image std (mean) | 0.188 | 0.164 |

Interesting nuance: the **old dataset's out-of-range excursions are less
universal but more severe when they occur** — fewer images are affected
below zero (61.5% vs 49.5%), but when a pixel does go out of range, a
larger fraction of that image's pixels are affected (2–3× the rate of the
new dataset). The new dataset's overshoot is more universal (99.1% of
images have at least one overshoot pixel, essentially all of them) but
milder per image. Both datasets confirm the same underlying fact from the
challenge brief — speckle noise pushes values outside [0,1] — just
distributed differently across images.

The old dataset's higher raw pixel variance (both GT and LR) is consistent
with it containing, on average, higher-contrast (if less fine-detailed —
see Section 3) content.

---

## 6. Frequency-domain view

`figures/05_radial_spectrum_compare.png` — radially-averaged power spectra
for both datasets, real NoisyLR vs. a clean reference downsample. Both
datasets show the same qualitative shape: a shared high-frequency rolloff
at low-to-mid frequencies (real image structure, nearly identical between
the "clean" and "noisy" curves — the downsampling operation is behaving the
same way in both datasets), then the noisy curve flattens out into a visible
noise floor at high frequencies while the clean curve keeps falling — this
is the frequency-domain signature of additive high-frequency noise sitting
on top of real signal, present in both datasets. The new dataset's overall
curves sit slightly above the old dataset's at most frequencies, consistent
with its higher-detail content (Section 3) carrying more energy across the
spectrum, not just at low frequencies.

---

## 7. Synthesis — what actually changed between the two data drops

1. **Same acquisition/degradation process** — same resolution, same 2×
   scale factor, same bicubic-like downsampling kernel, same fundamental
   speckle+Gaussian noise mixture. Nothing about *how* the data was
   generated appears to differ.
2. **Different, harder content mix** — the new dataset contains
   substantially more high-frequency, fine-detail imagery (2.2× higher
   average GT Laplacian variance), which is inherently harder to restore
   and interacts with the (also present in both) signal-dependent noise
   more severely.
3. **The new dataset is measurably harder and more uniformly hard** — 2.5 dB
   lower bicubic baseline, smaller spread (fewer easy outliers).
4. **Both datasets have a real, non-trivial rate of corrupted ground truth**
   (~0.6–1.0%) that should be excluded from training regardless of which
   dataset is in use — this isn't specific to the new drop.
5. **Noise-modeling assumptions transfer cleanly across both** — a
   heteroscedastic, signal-dependent noise model and a bicubic-kernel
   downsampling assumption are appropriate for both datasets; nothing here
   suggests the two datasets need fundamentally different modeling
   approaches, only that the new dataset demands more of whatever approach
   is used.

---

## 8. Files in this folder

```
dataset_comparison/
├── DATASET_COMPARISON_REPORT.md    this file
├── compare_datasets.py              the analysis script (re-runnable)
├── comparison_stats.json            full numeric results, both datasets
└── figures/
    ├── 00_old_corrupted_check.png              visual confirmation of corrupted-GT detection on the old dataset
    ├── 01_residual_distribution_compare.png     noise-only residual histograms, overlaid
    ├── 02_signal_dependence_compare.png         residual variance vs. intensity, overlaid (Experiment 1)
    ├── 03_degradation_dominance_compare.png     speckle-vs-Gaussian dominance score, overlaid (Experiment 2)
    ├── 04_bicubic_baseline_compare.png          bicubic-only PSNR/SSIM distributions, overlaid
    └── 05_radial_spectrum_compare.png           frequency-domain comparison
```
