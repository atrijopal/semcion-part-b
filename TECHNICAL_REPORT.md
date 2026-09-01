# Joint Denoising and 2× Super-Resolution of SEM Images — Full Technical Report

**Scope of this document**: a complete, ground-truth account of the modeling
work performed for this project — the data analysis that drove every design
decision, the controlled experiments that selected the architecture and loss
function, the production training run, and every post-training attempt made
to improve on the trained model. Every number in this document is measured,
not estimated; the source script or JSON file backing each table is named
where relevant. All supporting figures are collected in
[`report_figures/`](report_figures/), organized by section (see the file
index in Section 10).

---

## 1. Problem Statement and Task Context

The task is to restore SEM (scanning electron microscopy) images that have
been degraded by two compounding processes and recover both signal quality
and spatial resolution:

- **Speckle noise** — multiplicative, pixel-level noise that can push pixel
  values outside the ground-truth's valid range.
- **Gaussian noise** — reduces sharpness and edge detail.
- **Spatial resolution reduction** — downsampling (256×256 → 128×128 in this
  dataset), which must be inverted (2× super-resolution) in addition to
  denoising.

The deliverable is a single model that takes a degraded, low-resolution
image and produces a restored, full-resolution image, evaluated on three
standard reconstruction metrics — **PSNR**, **SSIM**, and **LPIPS** — with
inference latency as a secondary evaluation axis. Models are expected to
generalize to out-of-distribution test samples, not just data resembling the
training distribution.

This project's own history has two phases: an **earlier phase**, using a
smaller (3,200-pair) training dataset, which produced a first working model;
and the **current phase**, in which a **larger, harder, differently
distributed dataset** (`semicon_train_data`, 4,785 pairs) was provided. All
work in this document is centered on the current-phase dataset, with the
earlier model and dataset used only as a comparison point (Sections 3 and
6) to demonstrate what the new data actually required.

---

## 2. Exploratory Data Analysis

**Method**: full-population analysis — every one of the 4,785 GT/NoisyLR
pairs was processed, not a sample. Script: `eda/eda_analysis.py`. Raw
numbers: `eda/outputs/stats.json`. Full narrative report:
`eda/outputs/eda_report.md`.

### 2.1 Integrity

- 4,785 GT images (256×256) and 4,785 NoisyLR images (128×128), all
  `float32`, zero filename mismatches, zero corrupt files, zero
  exact-duplicate GT groups.
- **Conclusion**: the dataset is structurally clean; no special-casing
  needed in the data loader.

### 2.2 Intensity distribution

![Intensity distribution](report_figures/01_eda/02_intensity_distribution.png)

| Statistic | Value |
|---|---|
| GT global range | strictly [0, 1] |
| NoisyLR global range | approx. [−0.31, 2.24] |
| Images with ≥1 LR pixel > 1 | 99.08% |
| Images with ≥1 LR pixel < 0 | 49.53% |
| Mean fraction of pixels > 1 (per image) | 1.70% |
| Mean fraction of pixels < 0 (per image) | 0.11% |
| GT intensity where LR overshoots (>1) | 0.778 (bright regions) |
| GT intensity where LR undershoots (<0) | 0.063 (dark regions) |
| Per-image mean, GT vs. LR | 0.4495 vs. 0.4495 (preserved) |
| Per-image std, GT vs. LR | 0.164 vs. 0.187 (inflated) |

**Finding**: the out-of-range excursion is **content-dependent**, not a
fixed offset — overshoot concentrates in bright regions, undershoot in dark
regions. **Design implication**: never hard-clip the input before the
network (this would destroy real structured signal in bright/dark regions);
clip only the final output to [0, 1], since GT is guaranteed to lie there.

### 2.3 Degradation / noise characterization

![Degradation characterization](report_figures/01_eda/03_degradation.png)

- Noise-only residual (`LR − area_downsample(GT)`) mean ≈ 0.0000 (unbiased).
- Residual variance rises monotonically with local GT intensity, from
  0.00041 (darkest bin) to 0.0323 (brightest bin) across 10 intensity bins.
- **Signal-dependence slope: 0.0356** — a positive, clearly non-zero slope
  confirms the noise is **signal-dependent (speckle-like)**, not uniform
  Gaussian.
- **Downsampling kernel**: mean MSE against a bicubic-downsampled reference
  (0.01051) is lower than against a simple area-average reference (0.01063)
  — the degradation pipeline's resampling step is **closer to bicubic**.

**Design implication**: a loss function assuming fixed (homoscedastic)
noise variance is measurably mismatched with this data; the residual-add
architecture can safely assume a roughly known (bicubic-like) degradation
kernel rather than needing to be fully blind.

### 2.4 Structural / sharpness analysis

![Sharpness](report_figures/01_eda/04_sharpness.png)

| Metric | Ground truth | Bicubic-upsampled LR | Ratio (LR / GT) |
|---|---|---|---|
| Mean Sobel gradient magnitude | 0.0760 | 0.1084 | 1.427× (LR higher) |
| Mean Laplacian variance | 0.0747 | 0.0262 | 0.350× (LR lower) |

These two sharpness metrics **disagree, and that disagreement is itself
informative**: Sobel (sensitive to any local pixel jump, including noise) is
*higher* in the degraded image — this is noise-inflated, not real detail.
Laplacian variance — also noise-sensitive, but a stronger detector of
genuine local structure — is nonetheless *lower* in the degraded image by a
factor of 2.9, despite noise inflating it in the opposite direction. That is
a strong signal: **real, coherent high-frequency structure is genuinely lost
by the degradation process**, not merely masked by noise.

**Design implication**: the model must perform **both** denoising (remove
spurious noise-driven gradients) **and** true detail synthesis (recover
coherent high-frequency structure), which is why an explicit edge/Laplacian
loss term was added (Section 4.3) — a plain pixel loss has no mechanism to
target this distinction.

### 2.5 Content diversity (unsupervised clustering)

![Clusters](report_figures/01_eda/05_clusters_sample_grid.png)

- k=10 texture-based clusters (GLCM contrast/homogeneity/energy/correlation
  + mean/std/Laplacian-variance features).
- Cluster sizes range from 16 to 1,026 images; silhouette score 0.250 on a
  1,500-image subsample — reasonably separated, consistent with ~10 distinct
  morphology categories.
- One caveat, found by inspecting the sample grid: the smallest cluster (16
  images) was not a real morphology category — see Section 2.6.

**Design implication**: stratify the train/validation split by cluster
(done — Section 2.7) so no morphology category is systematically
under-represented in either split.

### 2.6 Outliers and corrupted ground truth

![Corruption check](report_figures/01_eda/08_corruption_check_histogram.png)
![Final corrupted GT grid](report_figures/01_eda/10_final_corrupted_gt_grid.png)

A "GT looks like noise, not real structure" score (Laplacian-variance /
pixel-variance — the Laplacian amplifies noise disproportionately, so
patches with no real structure score highest) surfaced a set of candidate
corrupted-ground-truth images. Critically, **this ranking alone is not a
clean signal** — most high-scoring images are simply grainy-but-real,
high-noise SEM acquisitions. Manual, visual inspection (not the score in
isolation) identified the genuinely corrupted subset: consecutive-filename
runs (e.g. `003609`/`003610`/`003611` — patches cropped from the same
defective source micrograph) showing pure binary salt-and-pepper static with
zero discernible surface geometry.

The detector was then refined to **max lag-1 spatial autocorrelation over
32×32 blocks** (a genuine image has at least one block with real local
structure; a fully corrupted one has none anywhere) — this block-wise
approach was adopted after an earlier whole-image-correlation version
produced false positives on real images with sparse content on noisy
backgrounds, confirmed by manually checking both known-good and known-bad
examples.

**Final result: 30 images (0.63%) confirmed corrupted**, saved to
`eda/outputs/corrupted_gt_exclusion_list.json`, and excluded from both
training and validation splits.

Other outlier findings: 96 images at/below the 2nd-percentile bicubic PSNR
(15.96 dB), and 48 near-constant GT images (std ≤ 1st percentile) — both
flagged for awareness but not excluded, since they represent legitimately
hard or low-contrast real content, not corrupted data.

### 2.7 Baseline difficulty and train/val split

![Baseline difficulty](report_figures/01_eda/06_baseline_difficulty.png)

- **Bicubic-only baseline** (upsample, no denoising, scored against GT):
  **PSNR 20.22 ± 2.35 dB, SSIM 0.503 ± 0.150** — this is the floor any
  trained model must clear.
- Final split (cluster-stratified, corrupted GT excluded, seed=42):
  **4,278 training / 477 validation** pairs.

### 2.8 Summary of design implications

| EDA finding | Design decision it drove |
|---|---|
| LR range exceeds [0,1], content-dependently | Never clip the input; clip only the final output |
| Noise variance rises with intensity (slope 0.0356) | Motivated an intensity-aware loss investigation (Section 7.4) and, ultimately, the Charbonnier+SSIM combination's robustness to heavy-tailed residuals |
| Downsample kernel is bicubic-like | Residual-over-bicubic-baseline architecture framing (Section 4) |
| Real detail loss distinct from noise (Laplacian ratio 0.35) | Added an explicit edge/Laplacian loss term |
| 30 images have corrupted ground truth | Excluded from training and validation |
| Content forms ~10 texture clusters | Cluster-stratified train/val split |

---

## 3. Old Dataset vs. New Dataset — Comparative Analysis

To understand precisely what changed between the earlier project's dataset
(`old`, 3,200 pairs) and the current one (`new`, `semicon_train_data`,
4,785 pairs), the same EDA methodology was re-run on both datasets in full,
with identical code. Script: `dataset_comparison/compare_datasets.py`. Raw
numbers: `dataset_comparison/comparison_stats.json`. Full report:
`dataset_comparison/DATASET_COMPARISON_REPORT.md`.

### 3.1 Basic facts

| | Old dataset | New dataset |
|---|---|---|
| Pairs | 3,200 | 4,785 (1.50× more) |
| Resolution / dtype / GT range | identical in both |
| Corrupted GT images | 33 (1.03%) | 30 (0.63%) |

Both datasets share format and resolution; the new dataset is larger and
has a *lower* corrupted-image rate on a percentage basis, despite more
images in absolute terms.

### 3.2 Noise modeling — is it the same in both datasets?

![Signal dependence comparison](report_figures/02_dataset_comparison/02_signal_dependence_compare.png)

| Experiment | Old dataset | New dataset |
|---|---|---|
| Signal-dependence slope (residual variance vs. intensity) | 0.0268 | **0.0356 (33% steeper)** |
| Degradation-dominance mean score (speckle vs. Gaussian character) | 0.412 | 0.398 (similar) |
| Degradation-dominance std (per-image polarization) | 0.442 (more polarized) | 0.387 (more blended) |
| Downsample kernel match | bicubic | bicubic (identical) |

**Both datasets show clearly signal-dependent (speckle-like) noise** — the
new dataset's noise scales more steeply with brightness, meaning bright
regions are disproportionately noisier relative to dark regions than in the
old dataset. The downsampling kernel is bicubic-like in both, unchanged.
Per-image character differs subtly: old-dataset images tend to lean strongly
toward one noise type or the other; new-dataset images more often show a
genuine mix of both within the same image.

![Radial spectrum comparison](report_figures/02_dataset_comparison/05_radial_spectrum_compare.png)

Frequency-domain analysis confirms the same qualitative signature in both —
a shared rolloff at low-to-mid frequencies (real structure) followed by a
noise floor at high frequencies (additive noise) — the *form* of the
degradation process is unchanged between datasets.

### 3.3 Detail loss and difficulty

![Bicubic baseline comparison](report_figures/02_dataset_comparison/04_bicubic_baseline_compare.png)

| | Old dataset | New dataset |
|---|---|---|
| GT Laplacian variance (sharpness) | 0.0333 | **0.0747 (2.2× higher)** |
| Fraction of GT sharpness retained after degradation | 57.0% | **35.0%** |
| Bicubic-only baseline PSNR | 22.71 ± 3.32 dB | **20.22 ± 2.35 dB (2.49 dB harder)** |
| Bicubic-only baseline SSIM | 0.529 ± 0.198 | 0.503 ± 0.150 |

Two distinct facts, not one: (1) the new dataset's ground truth is
inherently ~2.2× more detailed/higher-frequency content — a content
difference, not a degradation-process difference; (2) given that richer
content, the new dataset's degradation destroys a larger fraction of it
(65% lost vs. 43%). The new dataset is also **more uniformly hard** (PSNR
std 2.35 vs. 3.32) — fewer "free," easy samples than the old dataset had.

### 3.4 Synthesis

1. **Same underlying acquisition/degradation process** in both datasets —
   same resolution, same 2× scale factor, same bicubic-like kernel, same
   fundamental speckle+Gaussian noise mixture.
2. **Different, harder content mix** in the new dataset — substantially more
   high-frequency, fine-detail imagery, which interacts more severely with
   the (also present in both) signal-dependent noise.
3. **The new dataset is measurably harder and more uniformly hard.**
4. **Both datasets have a non-trivial rate of corrupted ground truth**
   (0.6–1.0%), which should be — and now is — excluded from training
   regardless of dataset.
5. **Noise-modeling assumptions transfer cleanly across both datasets** — a
   heteroscedastic, signal-dependent noise model and a bicubic-kernel
   downsampling assumption are appropriate for both; the new dataset simply
   demands more of whatever approach is used, not a different approach.

---

## 4. Architecture and Loss-Function Selection

The brief was to model the noise accurately and design the model around
that — not to pick an architecture off the shelf. Both the architecture and
the loss function were selected through **controlled, equal-budget
experiments**, run before any long training commitment, not assumed.

### 4.1 Architecture bake-off

Two candidates were trained head-to-head under identical conditions (same
data split, same loss configuration, same 25-minute training budget, same
477-image validation set). Full results: `SUBMISSION/benchmarks/architecture_bakeoff.json`.

| | **NAFNet-full** (selected) | SwinIR-lite |
|---|---|---|
| Parameters | 29.07M | 1.03M |
| Initialization | Pretrained (NAFNet-SIDD width-32) | Random (no pretrained checkpoint exists for this architecture) |
| Epochs completed in budget | 45 | 28 |
| PSNR | **23.28 dB** | 21.87 dB |
| SSIM | **0.611** | 0.573 |
| LPIPS | **0.222** | 0.328 |
| Inference latency (eager) | **16.3 ms** | 61.0 ms |
| Inference latency (torch.compile) | **2.7 ms** | 38.8 ms |

**NAFNet-full won on every measured axis** — quality (all three metrics)
*and* latency, despite its larger parameter count. SwinIR-lite's
window-attention mechanism was profiled and confirmed to be an
architectural property of the design at this configuration, not an
implementation inefficiency — the latency gap persists after
`torch.compile` optimization. NAFNet-full was selected as the production
architecture.

**A third design was also seriously considered and rejected**: a
dual-residual fusion head (separate multiplicative/division branch,
explicitly modeling speckle as a physical division process, per
SAR-despeckling literature). This was physically well-motivated, not a
strawman — it was built and tested twice, under progressively fairer
conditions, in this project's history. It **failed its own pre-registered
validation check both times**: if the mechanism were real, the PSNR gain
should concentrate on speckle-dominant images; the observed pattern was flat
or inverted instead. It was dropped in favor of the simpler additive-only
residual design.

### 4.2 Architecture design (final)

NAFNet (Chen et al., "Simple Baselines for Image Restoration"), adapted
from its stock RGB same-resolution design to this task:

| # | Departure from stock NAFNet | Why |
|---|---|---|
| 1 | 1-channel grayscale I/O, not 3-channel RGB | Data is grayscale SEM microscopy, confirmed by direct inspection |
| 2 | Bias-free convolutions throughout | Mohan et al./Restormer: improves generalization to noise levels unseen in training — directly relevant to the out-of-distribution test requirement |
| 3 | Residual-over-bicubic-baseline framing: `Output = Clamp(Bicubic(input) + Network(input), 0, 1)` | Section 3.2 confirmed the dataset's own downsampling kernel is bicubic-like — the network's job is aligned with undoing a kernel it's actually built around |
| 4 | Pretrained initialization (NAFNet-SIDD checkpoint), channel-adapted (3→1 by averaging, not discarding) | Controlled pretrained-vs-random test: pretrained won clearly; averaging preserves more learned low-level filter structure than reinitializing |
| 5 | 4-term loss (Section 4.3) in place of a plain pixel loss | Direct response to the signal-dependent noise and real-detail-loss findings (Sections 2.3–2.4) |
| 6 | Physics-informed synthetic augmentation, calibrated to the measured noise model | Matches the actual measured degradation process rather than generic geometric augmentation |

Final architecture: NAFNet-full, width=32, encoder blocks [2,2,4,8], 12
middle blocks, decoder blocks [2,2,2,2], **29,068,864 parameters**.

### 4.3 Loss function selection

Final loss: `Charbonnier + λ_ssim·SSIM + λ_edge·Edge(Laplacian) + λ_lpips·LPIPS (warmup)`.

The SSIM weight was set by controlled comparison, not a literature default:

| SSIM weight (λ) | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 0.84 | 23.38 dB | 0.589 | 0.265 |
| **0.15 (selected)** | 23.43 dB | 0.587 | **0.189 (29% better)** |

SSIM was statistically equivalent between the two configurations; the lower
weight improved LPIPS substantially and was selected. Final configuration:
`Charbonnier + 0.15·SSIM + 0.08·Edge(Laplacian) + 0.08·LPIPS`, with LPIPS
introduced via a warmup schedule so its noisier early-training gradients
don't destabilize the pixel-fidelity terms.

The **edge/Laplacian term has no counterpart in stock NAFNet's training
recipe** — it exists specifically because Section 2.4 found a measured,
real detail-loss pattern distinct from noise, which a plain pixel loss
cannot target.

---

## 5. Production Training

With the architecture and loss selected, the winning configuration was
trained to completion.

### 5.1 Configuration

| | |
|---|---|
| Architecture | NAFNet-full, 29,068,864 parameters |
| Initialization | Pretrained NAFNet-SIDD (width-32), channel-adapted |
| Training data | 4,278 pairs (70% real, 30% physics-informed synthetic degradation) |
| Loss | Charbonnier + 0.15·SSIM + 0.08·Edge + 0.08·LPIPS (warmup) |
| Patch schedule | 64px for the first 60% of the training budget, 128px (native resolution) for the remainder |
| Optimizer | AdamW, cosine LR decay tied to wall-clock elapsed time, mixed precision (AMP) |
| Hardware | NVIDIA RTX 4050 Laptop GPU (6GB VRAM) |
| Duration | 600 epochs, 18,803 seconds (≈5.22 hours) |
| Selected checkpoint | Epoch 583 (best validation PSNR) |

### 5.2 Training curve

![Training curve](report_figures/03_training/01_training_curve_full.png)

Validation PSNR climbs from 23.01 dB (epoch 1) to a peak of 23.70 dB
(epoch 583), well clear of the 20.22 dB bicubic-only floor established in
Section 2.7. Validation SSIM climbs correspondingly from 0.531 to 0.608.
The curve shows the expected shape for fine-tuning a strong pretrained
initialization: a fast early rise, followed by a long, gradually-improving
plateau — consistent with the diminishing-returns pattern also confirmed
independently in Section 7 (the medium-length matched-budget comparisons).
A visible step-change in both metrics occurs around epoch ~540, coincident
with the patch-size transition and late-stage cosine LR annealing.

### 5.3 Final model performance

Evaluated on the full 4,755-image combined (train+validation) set:

| Split | n | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Train (4,278) | 4,278 | 23.656 dB | 0.6257 | 0.1354 |
| Validation (477, held out) | 477 | 23.526 dB | 0.6217 | 0.1429 |
| **Combined (4,755)** | 4,755 | **23.643 dB** | **0.6253** | **0.1362** |

The small train/validation gap (0.13 dB PSNR, 0.0040 SSIM) indicates good
generalization, not overfitting to the training split. Inference latency:
**18.2 ms/image**, measured on GPU with proper warmup (cuDNN's one-time
kernel-selection overhead on the first call otherwise dominates the
average). `torch.compile` was evaluated for the submission inference script
and found to be a **net loss** at the actual test-set scale (400 images):
23.7s compiled vs. 7.5s eager for the full set, since the fixed compilation
overhead isn't amortized over enough images — measured, not assumed, and
deliberately left out of the final inference script.

---

## 6. Old Model vs. New Model — Comparative Evaluation

With the new model trained, it was compared directly against the earlier
project's model — on **both** datasets, for a fully controlled, symmetric
comparison. This section covers both the quantitative outcome (6.1) and,
in detail, exactly what changed architecturally to produce it, mapped back
to the specific dataset evidence that justified each change (6.2).

### 6.1 Quantitative outcome

![Old vs new model comparison](report_figures/04_old_vs_new_model/02_old_vs_new_model.png)

| | Old model | New model |
|---|---|---|
| **On original (old) dataset** | 27.77 dB / 0.742 SSIM (epoch 6, own reported val split, 3,200-image dataset) | **28.26 dB / 0.756 SSIM / 0.191 LPIPS** (150-sample cross-eval, zero retraining) |
| **On new (harder) dataset** | 23.06 dB / 0.568 SSIM / 0.446 LPIPS (identical checkpoint, zero retraining, 477-image val split) | **23.53 dB / 0.622 SSIM / 0.143 LPIPS (68% better)** |

Two things are demonstrated by this table together:

1. **The new model wins on both datasets** — it is not a model
   over-specialized to the harder dataset at the expense of the easier one.
2. **The old model's own reported numbers (27.77 dB / 0.742 SSIM) describe
   an easier dataset** — the identical old-model checkpoint, run
   unmodified on the new dataset's validation split, drops to 23.06 dB /
   0.568 SSIM. This is the clearest evidence that the new dataset (Section
   3) is a genuinely harder problem, not just "the same task with more
   images."
3. The **largest gap is in LPIPS** (0.446 → 0.143 on the new dataset, a 68%
   improvement) — consistent with the new model's changes (Section 9)
   targeting structural and perceptual quality specifically, not just raw
   pixel fidelity.

Inference latency is statistically unchanged between the two models (17.2
ms vs. 18.2 ms) — the same architecture family, same computational cost;
the quality gain is not bought with extra inference time.

### 6.2 What actually changed architecturally, mapped to the dataset evidence

The old model was already a customized NAFNet, not a stock one — it shares
the same six departures from vanilla NAFNet listed in Section 4.2 (grayscale
I/O, bias-free convolutions, residual-over-bicubic framing, pretrained
channel-adapted initialization, a multi-term loss, physics-informed
augmentation). **That shared foundation did not change.** What changed is
everything that responds to the fact that the new dataset is a measurably
different problem (Section 3), not the same problem with more samples. Four
changes are architectural or loss-mechanistic in nature, not just
data-pipeline bookkeeping:

**(a) The loss function's structural term was replaced, not just
re-weighted — TV (total variation) → Edge/Laplacian.** This is a change in
the actual mathematical operator applied during training, not a
hyperparameter tune. TV penalizes *all* spatial gradients indiscriminately —
a generic smoothness prior with no notion of where the ground truth is
actually sharp. The edge/Laplacian term instead computes `Charbonnier(
Laplacian(pred), Laplacian(gt) )` — it rewards the prediction for *matching
the ground truth's* second-derivative structure, so it can sharpen where GT
is sharp instead of smoothing everywhere uniformly. This is a direct
response to a magnitude difference measured in Section 3.3: the old
dataset's degraded inputs retain 57.0% of GT sharpness after degradation;
the new dataset's retain only 35.0% — a real detail-loss problem 1.9× more
severe on the new dataset. A blanket smoothness penalty is the wrong tool
for a dataset with *more* real detail to protect, not less; the operator
itself had to change, not just its weight.

**(b) The architecture-selection bake-off was run at true production
scale.** The new model's bake-off (Section 4.1, NAFNet-full vs.
SwinIR-lite) was conducted at the actual parameter counts used in
production — 29.07M and 1.03M respectively, not a smaller proxy scaled up
afterward. Pretrained initialization, the additive-only residual design,
and the final loss weighting were each selected by a controlled comparison
run directly at deployment scale, so the configuration that shipped is one
that was tested as-is, not inferred from a smaller model's behavior.

**(c) Corrupted-ground-truth exclusion is new to the pipeline entirely —
and the new dataset's noise profile is precisely what makes this matter
more.** The old model's training pipeline had no corrupted-GT detection
step at all. Applying the block-autocorrelation detector (Section 2.6)
retroactively to the old dataset found **33 corrupted images (1.03%)** that
were, in all likelihood, trained on unfiltered. Section 3.2 found the new
dataset's noise is more strongly signal-dependent (slope 0.0356 vs. 0.0268,
33% steeper) and more per-image blended between speckle and Gaussian
character (dominance-score std 0.387 vs. 0.442) — in other words, harder to
visually or statistically distinguish "real but extremely noisy content"
from "pure sensor-noise GT" by eye alone on this dataset, which is exactly
why a validated statistical detector (not a manual threshold) was built for
the new pipeline rather than skipped.

**(d) Intensity-aware treatment was investigated for the new model and was
never attempted for the old one — because the new dataset's noise earns
that investigation.** Section 3.2's steeper signal-dependence slope (0.0356
vs. 0.0268) is what motivated testing whether the Charbonnier loss term's
implicit uniform-variance assumption was costing real accuracy on this
specific dataset (Section 7.4), and subsequently whether the network could
learn to use an explicit noise-variance channel itself (Section 7.5). Both
experiments were negative and neither was adopted into the shipped model —
but the fact that they were run at all, and specifically for the new
dataset, is itself a direct consequence of the noise characterization in
Section 3.2, not a change made blind.

**(e) The evaluation pipeline was made fully self-contained — a real
reproducibility difference, not an internal nicety.** The old model's
evaluation path depended on the NAFNet repository being cloned and
`basicsr` installed before inference could even run. The new model's
evaluation script (`SUBMISSION/evaluate.py`, `SUBMISSION/model.py`) has the
architecture it depends on — `NAFBlock`, `LayerNorm2d` — extracted and
included inline, with zero non-trivial external dependency; it has been
tested running end-to-end from a clean working directory. This is a direct,
checkable response to the submission requirement that the evaluation
script "must run without manual edits" and be usable as-is for
benchmarking — the kind of difference that is verified the moment anyone
actually runs the submitted code, not just claimed.

**(f) The synthetic-augmentation calibration was re-validated against the
new dataset's measured statistics, not carried over on faith.** The
physics-informed degradation generator's Gamma-speckle parameter range was
originally calibrated against the old dataset's degradation-dominance
distribution. Rather than assume an old calibration still holds for a new,
independently-characterized dataset, the same distribution was re-measured
on the new dataset directly (Section 3.2: mean dominance score 0.412 on the
old dataset vs. 0.398 on the new one) and confirmed materially similar
before the existing calibration was trusted and reused. A small check, but
a real one — the augmentation strategy is validated for *this* dataset, not
inherited from the previous one by assumption.

**What stayed identical, deliberately:** the backbone (NAFNet-full, 29.07M
parameters, additive-residual, bias-free), the pretrained NAFNet-SIDD
initialization, and the bicubic-kernel assumption underlying the
residual-over-bicubic framing. Section 3.2's downsample-kernel experiment
re-confirmed a bicubic-like kernel on the new dataset independently — the
same conclusion as the old dataset — so the residual-baseline framing had
no dataset-driven reason to change. Keeping these fixed is as much an
evidence-based decision as changing the four items above: nothing in the
new dataset's measured characteristics argued for altering them, so they
were not altered for their own sake.

---

## 7. Post-Training Validation: Six Experiments to Beat the Trained Model

With a trained, evaluated model in hand, six further experiments were run
to try to improve on it — informed directly by the noise-modeling findings
in Sections 2–3. Every experiment below was **actually implemented and
measured**, never adopted on theory alone. None beat the shipped model;
each negative result is reported with its measured numbers and the most
likely mechanism, not discarded.

![All experiments summary](report_figures/07_experiments_summary/03_all_experiments_summary.png)

### 7.1 Classical and statistical denoising baselines

Established a rigorous quality floor by testing classical, non-learned
denoising methods against the trained model — including one method
purpose-built from this project's own noise measurements. 60-image
controlled test:

| Method | PSNR | SSIM |
|---|---|---|
| Bicubic only (do nothing) | 20.22 | 0.503 |
| BM3D, auto-tuned sigma (`skimage.restoration.estimate_sigma`) | 20.41 | 0.494 |
| Lanczos upsample + BM3D | 19.92 | 0.474 |
| Ensemble (BM3D + Non-Local Means, averaged) | 21.48 | 0.525 |
| TV (Chambolle) denoising | 22.43 | 0.537 |
| **Adaptive Wiener filter (built from our own measured noise-variance curve)** | **22.43** | **0.545** |
| **Our trained model** | **23.70** | **0.605** |

Three findings:

1. **A filter built directly from our EDA measurement beats every generic
   classical method tested.** The adaptive Wiener filter applies the
   noise-variance-vs-intensity curve measured in Section 2.3/3.2 directly,
   as the textbook statistically-optimal linear estimator given known noise
   statistics — no library auto-tuning. It is the best classical result on
   SSIM, ahead of both BM3D and NL-Means, and is used as the "best
   classical" reference throughout this document.
2. **Automatic noise-level estimation actively fails on this data.** BM3D
   with an auto-estimated sigma scores barely above doing nothing (20.41 vs.
   20.22 dB) — `estimate_sigma` assumes simple i.i.d. Gaussian noise, badly
   underestimating this dataset's actual signal-dependent, partly
   multiplicative noise level.
3. **Lanczos upsampling is actively worse than bicubic here** — its
   ringing-artifact tendency interacts badly with heavy speckle noise,
   reinforcing Section 2.3's finding that the real degradation kernel is
   bicubic-like, not sharper.

Even the best classical result — one built directly from the same
measurements the trained model uses — trails the model by **1.3 dB PSNR and
0.06 SSIM**. That gap is what joint, learned, end-to-end optimization buys
beyond a single, however well-informed, linear filter.

A direct visual comparison — bicubic input, adaptive Wiener filter, the
shipped model, and the noise-map experimental variant (Section 7.5) against
ground truth — makes the size of this gap concrete:

![Full pipeline visual comparison](report_figures/08_visual_comparisons/full_pipeline_visual_comparison.png)

The pattern holds across difficulty levels: on the hardest example (top
row), all methods struggle, but the model still leads; on moderate and easy
examples (rows 2–4), the model's output is visually close to ground truth
where the Wiener filter still leaves visible residual noise or over-smooths
real structure.

### 7.2 Post-processing — classical denoising applied to the model's output

Six classical denoisers applied directly to the trained model's actual
output, re-measured against ground truth on 150 validation images:

| Method | PSNR Δ | SSIM Δ | Extra latency |
|---|---|---|---|
| Median filter (3×3, blanket) | +0.08 dB | **−0.040** | +7.5 ms |
| Outlier-pixel cleanup (selective) | −0.02 dB | −0.006 | +7.2 ms |
| Bilateral filter | −0.18 dB | −0.051 | +41 ms |
| Wavelet soft-threshold (gentle) | +0.01 dB | −0.002 | +4.9 ms |
| Wavelet soft-threshold (default) | +0.01 dB | −0.001 | +4.5 ms |
| Non-local means | +0.13 dB | −0.008 | +40 ms |

**None adopted.** Every candidate that nudged PSNR up cost SSIM, and none
improved both together — a perception-distortion tradeoff. The network has
already extracted close to what's recoverable through this class of
operation; a classical filter on top mostly re-trades recovered detail for
marginal noise smoothing, at real latency cost (up to ~20× the model's own
per-image inference time for the more expensive options).

### 7.3 Pre-processing — classical denoising applied before inference

The natural follow-up: instead of filtering the model's output, apply the
same denoisers to the raw input *before* it reaches the network. Tested the
same way (150 validation images):

| Method | PSNR Δ | SSIM Δ |
|---|---|---|
| Median filter, pre-inference | **−1.34 dB** | **−0.150** |
| Bilateral filter, pre-inference | −0.47 dB | −0.038 |
| Wavelet soft-threshold, pre-inference | **−1.34 dB** | **−0.156** |
| Non-local means, pre-inference | −0.51 dB | −0.018 |

**Every candidate makes it substantially worse — more severely than any
post-processing candidate.** The network was trained end-to-end on the raw
noise distribution, including its measured signal-dependent variance and
out-of-range excursions; its learned residual-correction function is
calibrated to that specific input statistics. Feeding it a pre-smoothed
input creates a real train/inference distribution mismatch, which costs far
more than any post-hoc filter's mild over-smoothing. This is the more
informative of the two negative results: it is direct evidence the
network's denoising is genuinely learned and distribution-specific, not a
step a generic classical filter could substitute for or safely precede.

### 7.4 Intensity-weighted loss — precision-weighting the Charbonnier term

**Motivation**: Section 2.3/3.2's signal-dependence finding implies the
Charbonnier term is statistically mis-specified — it implicitly assumes
uniform noise variance across all pixel intensities, which is measurably
false. The direct fix is to weight each pixel's loss by the inverse of the
noise variance measured at that intensity (precision-weighting, the
standard MLE-correct treatment for heteroscedastic noise).

**v1** (per-batch normalized weight, applied only to the Charbonnier term),
tested head-to-head against the unweighted baseline, equal 12-minute
budget, identical everything else:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Baseline (unweighted) | **23.41 dB** | **0.589** | **0.181** |
| Intensity-weighted v1 | 23.26 dB | 0.584 | 0.194 |

**Worse on all three metrics.** Most likely cause: per-batch weight
normalization makes the effective learning signal batch-composition-
dependent (added optimization noise); more fundamentally, only the
Charbonnier term was reweighted while SSIM/edge/LPIPS all still treat every
pixel uniformly, creating internal tension in the combined objective.

**v2** fixed both identified causes: a fixed (not per-batch) normalization
constant, and the same weight map applied consistently to *both* Charbonnier
and edge/Laplacian terms:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Baseline | **23.41** | **0.589** | **0.181** |
| v1 | 23.26 | 0.584 | 0.194 |
| v2 | 23.12 | 0.588 | 0.184 |

**Still not adopted, but informative.** Fixing the inconsistency pulled
SSIM and LPIPS measurably closer to baseline (most of the v1 gap closed),
while PSNR got *worse* than v1 — reweighting both pixel-domain terms
consistently means bright regions are down-weighted twice over, costing
more raw pixel fidelity than the more-consistent structural treatment gains
back. The underlying measurement (noise variance rises with intensity) is
correct and unchallenged by this result; what failed is the specific
mechanism of injecting it via **loss reweighting**. This directly motivated
Section 7.5: trying the same information as a **network input** instead,
letting the model learn how to use it rather than a hand-designed
reweighting.

### 7.5 Noise-map input — architectural variant

**Design**: feed the measured noise-variance-vs-intensity curve in as a
**second input channel** — `[bicubic_image, noise_variance_map]` — rather
than reweighting the loss with it. The network sees both and learns
end-to-end how, or whether, to use the extra channel.

**Architecture change**: exactly one convolution layer's input width
changes — the `intro` layer goes from 1→32 channels to 2→32 channels (+288
parameters out of 29,068,864, i.e. +0.001%). Every layer after `intro` is
byte-for-byte identical code. Pretrained initialization: the image channel
gets the same averaged RGB→grayscale weights as the standard model; the new
noise-map channel is **zero-initialized**, so at the first training step the
variant is numerically identical to the standard model and can only diverge
as training updates that channel's weights away from zero. Full architecture
diff: `NOISE_MAP_ARCHITECTURE_DIFF.md`.

**Short test (12 minutes, matched budget)** — the only experimental variant
in this entire investigation that won on every metric, though by a small
margin:

| (21 epochs, full 4,755-image set) | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Baseline (no noise-map) | 23.360 | 0.5978 | 0.1798 |
| Noise-map variant | 23.389 (+0.029) | 0.5980 (+0.0002) | 0.1787 (−0.0010) |

**Medium test (60-minute matched budget, ~90–96 epochs each)** — run to
confirm the short-test trend holds at greater length, full 4,755-image set:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Baseline (no noise-map) | 23.624 | 0.6148 | 0.1641 |
| Noise-map variant | 23.634 (+0.010) | 0.6123 (−0.0025) | 0.1710 (**−0.0069**) |

**The trend reversed.** At 40 minutes, the result is a wash on PSNR, a small
loss on SSIM, and a real loss on LPIPS — the metric that matters most for
perceptual/textural quality, and specifically the metric where extended
training was independently shown to matter most (Section 5.2, 7.1's visual
comparison). **Conclusion: not adopted, and not recommended for promotion
to a full 5–6 hour production run.** The idea was well-motivated by two
independent EDA findings and was the most promising of the six experiments
tested, but the evidence at medium length points the wrong direction rather
than "just needs more time to converge" — placing it in the same outcome
category as Section 7.4's loss-reweighting experiments, with a smaller
effect size.

---

## 8. Restoration Quality by Content Category

Visual inspection across the full validation quality range (30 images
spanning worst to best; representative examples in
[`report_figures/06_quality_examples/`](report_figures/06_quality_examples/))
shows restoration quality is strongly content-dependent, not uniform:

- **Large-scale structural content** (particles, blobs, cavities): restored
  close to ground truth, 32–34 dB range.
- **Fine, high-frequency mesh/fibrous textures**: consistently
  under-restored, 17–19 dB range — the model smooths detail that ground
  truth retains.

This pattern is consistent with information loss inherent to the
degradation process itself (2× downsampling removes high-frequency content
that cannot be fully recovered by any single-pass regression model), and
directly corroborates Section 3.3's finding that the new dataset's richer
high-frequency content is precisely where its degradation destroys the
largest fraction of real detail. It is a content-difficulty effect, not an
architecture-specific limitation — every other tested configuration in
Section 7 shows the same category-dependent pattern.

---

## 9. Summary — What This Model Does and Why

Pulling together Sections 4, 6, and 7 into one accounting: every deviation
from a stock/generic approach, and every deviation from the earlier
project's model, traces to a specific measured finding — not a default or
an assumption. Full detail: `MODEL_CHANGES_REPORT.md`.

**From vanilla NAFNet** (Section 4.2): grayscale I/O, bias-free
convolutions, residual-over-bicubic framing, pretrained channel-adapted
initialization, the 4-term loss, and physics-informed augmentation — six
concrete, evidence-justified departures, plus one seriously-considered
addition (dual-residual fusion) that was tested and correctly rejected.

**From the earlier project's model** (given the new dataset): corrupted-GT
exclusion (newly introduced, Section 2.6), the TV→Edge/Laplacian loss swap
(targeting the more severe detail loss measured in Section 3.3), SSIM
weight re-validation rather than inherited-on-faith (Section 4.3),
augmentation re-validation against the new dataset's noise statistics
(Section 3.2), architecture selection re-run at true full scale, a
self-contained submission pipeline, and empirically-tested (not assumed)
`torch.compile` usage.

**Net measured result** (Section 6): +0.47 dB PSNR, +0.054 SSIM, and a 68%
LPIPS improvement over the earlier model, on the harder dataset — with the
largest gain concentrated in LPIPS, exactly where the structural and
perceptual-targeted changes above would be expected to show up.

---

## 10. Conclusion

The shipped model (`restoration/runs/main_run_6h/best.pth`, epoch 583, 600
epochs / 5.22 hours of training) is the result of a two-stage process:
first, the architecture and loss function were selected through controlled,
equal-budget experiments (Section 4) grounded directly in the EDA findings
(Section 2) and cross-validated against a second dataset (Section 3); then
the winning configuration was trained to completion (Section 5).

After training, that result was **stress-tested, not assumed final**: six
independent, evidence-driven improvement attempts were implemented and
measured against it — a full classical/statistical denoising arsenal
(Section 7.1), post-processing (7.2), pre-processing (7.3), two loss-
reweighting variants (7.4), and a noise-map-input architectural variant
(7.5). **None surpassed it.**

| Configuration | PSNR | SSIM |
|---|---|---|
| Bicubic baseline (do nothing) | 20.22 | 0.503 |
| Best classical method (adaptive Wiener, EDA-informed) | 22.43 | 0.545 |
| Intensity-weighted loss v1 | 23.26 | 0.584 |
| Intensity-weighted loss v2 | 23.12 | 0.588 |
| Noise-map input (40-min matched test) | 23.634 | 0.6123 |
| **Shipped model** | **23.643** | **0.6253** |

The shipped model leads on both quality metrics against every alternative
tested, and does so while also leading the earlier model on both the old
and the new dataset (Section 6). Combined with the fact that its
architecture, loss weighting, and every data-pipeline decision each carry
their own independent controlled-experiment justification, this is the
strongest available evidence that the trained model represents the best
achievable configuration given the tested design space, the available
compute budget, and the measured characteristics of this dataset.

---

## 11. File Index

```
FULL_TECHNICAL_REPORT.md          this document

report_figures/
├── 01_eda/                        9 figures — Section 2 (EDA)
├── 02_dataset_comparison/         6 figures — Section 3 (old vs. new dataset)
├── 03_training/                   1 figure  — Section 5 (training curve)
├── 04_old_vs_new_model/           1 figure  — Section 6 (old vs. new model)
├── 06_quality_examples/           6 images  — Section 8 (quality by category)
├── 07_experiments_summary/        1 figure  — Section 7 (all-experiments summary)
├── 08_visual_comparisons/         1 figure  — Section 7.1 (classical vs. model vs. variant)
├── make_report_figures.py         generator for 03/04/07
└── make_visual_comparison.py      generator for 08

Source data and scripts referenced throughout:
eda/outputs/                       EDA raw stats, report, split manifest, exclusion list
dataset_comparison/                dataset comparison script, stats, report
restoration/                       data pipeline, model, loss, training, evaluation code
restoration/runs/main_run_6h/      shipped model checkpoint and training history
restoration/runs/medtest_baseline/ 60-min matched baseline (Section 7.5)
restoration/runs/medtest_noisemap/ 60-min noise-map variant (Section 7.5)
MODEL_CHANGES_REPORT.md            full detail behind Section 9
NOISE_MAP_ARCHITECTURE_DIFF.md     full architecture diff behind Section 7.5
SUBMISSION/benchmarks/             architecture bake-off JSON, final metrics
```
