# Milestone 2 Report: Segmentation Pipeline, First Submissions and Post-Processing Analysis

**Project:** SurfaceGuard AI, industrial surface anomaly segmentation (Deep Learning & GenAI Project, T3-2026)
**Student:** Surya Prakash Pareek, roll no. 22f2001613
**Branch:** `milestone-2` | **Date:** 26 Oct 2026
**W&B project:** `22f2001613-indian-institute-of-technology-madras / 22f2001613-t32026`

---

## 1. Summary

Milestone 2 turned the exploratory work of Milestone 1 into a working segmentation pipeline: a PyTorch Dataset and DataLoader, two U-Nets trained from scratch, W&B tracking, conversion of predictions to run-length encoding (RLE), and four Kaggle submissions. The main results:

| Submission | Model | Post-processing | Leaderboard score |
|---|---|---|---|
| Dummy | all masks empty | none | 0.27337 |
| 1 | UNetScratch (epoch 14) | threshold 0.8 | 0.49780 |
| 2 | UNet3Stage (Q8, epoch 4) | threshold 0.5 | 0.45898 |
| 3 | UNetScratch + flip TTA | per-class threshold, minimum region size and confidence gate | **0.56021** |

Key findings:

1. **Over-prediction was the main loss.** Raw predictions flagged a defect on about 45% (a), 39% (b), 54% (c) and 69% (d) of test image-class pairs, against true rates of 9.5%, 2.7%, 36% and 15%. Per-class post-processing together with flip test-time augmentation (Section 7) lifted the leaderboard score from 0.4978 to 0.5602 without retraining.
2. **The leaderboard formula is not exactly our helper metric.** A per-pair reading (any predicted pixel on an empty pair counts as a failure) tracks the leaderboard much better than an area-based reading (Section 3.3).
3. **Remaining headroom is in detection and mask quality, not in false-positive control.** Even perfect handling of empty pairs would only give 0.648 with the current masks (Section 7.4). The model finds 77% of real defects and its masks reach Dice 0.65 when it finds them.
4. **Decision:** scratch models are not worth further investment. Milestone 3 moves to pretrained encoders, reusing the pipeline, the tracking and the analysis tooling built here.

---

## 2. Task and Data Recap

- **Data:** 11,393 training images (128 x 800, grayscale line-scan strips) and 3,000 test images. `train.csv` is in long format: 45,572 rows = 11,393 images x 4 defect classes. An empty mask is NaN in `train.csv` and a single space `" "` in `sample_submission.csv`.
- **Class imbalance:** images containing each defect: a 1,081, b 313 (rarest), c 4,100 (most common), d 1,707. About 84.2% of image-class pairs are empty, and defect pixels are about 0.77% of all pixels. 5,516 images are completely clean.
- **Co-occurrence and overlap:** 1,198 images contain more than one defect type. Defect classes co-occur at image level, but their masks never share a pixel, so the classes are spatially exclusive. The strong raw co-occurrence of `defect_c` and `defect_d` is largely explained by how common `defect_c` is; the correlation analysis in the questions notebook normalises for this base-rate effect.
- **Split:** stratified 80/20 split by image over 15 label-combination strata (seed 42): 9,114 training and 2,279 validation images (`data/holdout.csv`). The same split is used for every model and analysis.
- **RLE convention:** 1-based start index, column-major (Fortran) traversal, space-separated `start length` pairs. The native 128 x 800 shape is kept (resizing to a square would distort the aspect ratio by 6.25x).

## 3. Method

### 3.1 Pipeline design decisions

| Decision | Choice | Alternatives considered | Reason |
|---|---|---|---|
| Output formulation | One network with 4 output channels (multi-channel joint prediction, "Paradigm 1") | Class-conditioned network (one forward pass per class, "Paradigm 2") | One forward pass per image; no 4x inference cost. Paradigm 2 would balance classes more easily and is a possible later experiment. |
| Output activation | 4 independent sigmoids | Softmax with a background class | Classes can co-occur in an image; sigmoids also handle any overlap. Since masks never overlap, softmax is valid too and is to be compared in Milestone 3. |
| Resolution | native 128 x 800 | resize to a square | resizing distorts thin defects; both sides are divisible by 8/16, so U-Nets accept the native size. |
| Loss | BCE-with-logits + soft Dice on image-class pairs that contain a defect | BCE only | with 84% empty pairs, BCE alone is minimised by predicting nothing; the Dice term rewards overlap. |
| Optimiser | AdamW | SGD | adaptive step sizes, decoupled weight decay, robust default learning rate. |
| Experiment tracking | Weights & Biases, key in Kaggle Secrets | local logs | required by the course; run names encode the experiment. |

### 3.2 Dataset

Two Dataset variants were built, because the main pipeline and the milestone question prescribe different inputs:

- **Main pipeline (scratch model):** grayscale, 1 channel, normalised as `(pixel/255 - 0.3314) / 0.1892`; augmentation on training images only (horizontal and vertical flips with probability 0.5, applied to image and masks together; contrast factor U(0.8, 1.2); brightness shift U(-0.1, 0.1); clip to [0, 1]).
- **Question Q7 / Q8 pipeline:** 3-channel RGB (OpenCV BGR to RGB; for these grayscale files this repeats the gray channel), values divided by 255 only, no augmentation. Masks are a `(4, 128, 800)` float32 tensor. Targets must be float32 because `BCEWithLogitsLoss` computes its terms in floating point and requires the target to have the logits' dtype; integer or boolean targets raise an error and cannot carry soft labels.

### 3.3 Evaluation metric

The official score is 0.70 x foreground Dice (on positive pairs) + 0.20 x false-positive score + 0.10 x empty-prediction accuracy. Our helper implements the area-based false-positive score (**reading A**): `1 - mean(predicted area / canvas)` on empty pairs. Because the leaderboard did not agree with it (an all-empty dummy submission scored 0.27337, while the helper gives 0.30), a second **reading B** was added: `0.7 x foreground Dice + 0.3 x empty accuracy`, where any predicted pixel on an empty pair counts as a failure.

Comparison on the validation set, using the two raw submissions with known thresholds:

| Submission | val A | val B | leaderboard | A - LB | B - LB |
|---|---|---|---|---|---|
| scratch, raw, tau 0.8 | 0.6028 | 0.4930 | 0.4978 | +0.105 | -0.005 |
| q8, raw, tau 0.5 | 0.5853 | 0.4985 | 0.4590 | +0.126 | +0.040 |

Reading B is much closer than A, but neither is exact; the formula remains partly unresolved. The final submission was a prediction test: B (held-out) predicted 0.584, A plus the constant offset predicted about 0.52, the leaderboard gave 0.560. Reading B is therefore used as the working proxy, expecting it to be optimistic by about 0.02 to 0.04.

## 4. Models

| | UNetScratch (main) | UNet3Stage (Q8) |
|---|---|---|
| Encoder levels | 4 (32, 64, 128, 256) + 512 bottleneck | 3 (32, 64, 128) + 256 bottleneck |
| Parameters | 7,762,564 | 1,927,108 |
| Input | 1 channel, mean/std normalised | 3 channels, /255 |
| Augmentation | flips, brightness, contrast | none |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | AdamW, lr 3e-4, weight decay 1e-4 |
| Epochs / batch | 15 / 16 | 5 / 16 |
| Time per epoch (Kaggle GPU) | about 372 s | about 283 s (T4) |
| Checkpoint rule | best validation score at tau 0.5 (epoch 14) | lowest validation loss (epoch 4) |
| W&B run | `m2-unet-scratch-v1` (id `34c5iose`) | `m2-q8-unet-3stage-5ep` (id `w6rle3pg`) |

**Architecture (both):** a U-Net. The encoder (repeated Conv3x3 - BatchNorm - ReLU pairs with max-pooling) shrinks the image and learns what is present; the decoder (transposed convolutions) restores the resolution to say where; skip connections concatenate encoder feature maps into the decoder to recover fine detail lost by pooling. A 1x1 convolution produces 4 logits per pixel; the sigmoid is applied inside the loss and at prediction time.

## 5. Training Results

### 5.1 UNetScratch (validation; score = reading A at tau 0.5)

| Epoch | Train loss | Val loss | Score | FG Dice |
|---|---|---|---|---|
| 1 | 0.9243 | 0.7363 | 0.4831 | 0.3591 |
| 2 | 0.6923 | 0.6817 | 0.5002 | 0.3948 |
| 3 | 0.6500 | 0.6228 | 0.5390 | 0.4671 |
| 4 | 0.6317 | 0.6151 | 0.5554 | 0.4484 |
| 5 | 0.6119 | 0.6117 | 0.5486 | 0.4529 |
| 6 | 0.6032 | 0.5970 | 0.5583 | 0.4776 |
| 7 | 0.5881 | 0.5806 | 0.5657 | 0.4904 |
| 8 | 0.5814 | 0.5803 | 0.5652 | 0.4704 |
| 9 | 0.5748 | 0.5582 | 0.5854 | 0.5009 |
| 10 | 0.5676 | 0.5577 | 0.5768 | 0.5029 |
| 11 | 0.5599 | 0.5513 | 0.5833 | 0.5103 |
| 12 | 0.5476 | 0.5273 | 0.5949 | 0.5245 |
| 13 | 0.5416 | 0.5518 | 0.5841 | 0.5132 |
| **14** | 0.5368 | 0.5311 | **0.6020** | 0.5233 |
| 15 | 0.5300 | 0.5365 | 0.5923 | 0.5211 |

Train and validation loss fall together and both were still falling at epoch 15: the model is **underfitting, not overfitting**. Validation is at or below training loss in many epochs because the training loss is averaged while the weights change during the epoch, with augmentation on and BatchNorm in training mode. The lowest validation loss was at epoch 12 (0.5273), while the checkpoint kept is epoch 14, selected on the validation score because the score, not the loss, is what is graded.

Threshold sweep with the epoch-14 checkpoint (validation, no other post-processing):

| tau | Score | FG Dice | Empty accuracy |
|---|---|---|---|
| 0.3 | 0.6007 | 0.5262 | 0.348 |
| 0.5 | 0.6020 | 0.5233 | 0.379 |
| 0.8 | 0.6028 | 0.5145 | 0.443 |
| 0.9 | 0.5997 | 0.4996 | 0.511 |

The curve is flat under reading A (a change of 0.003 is within noise), but empty accuracy rises steadily with the threshold; this is the first sign that false detections matter.

### 5.2 UNet3Stage (Q8)

| Epoch | Train loss | Val loss | Score@0.5 | FG Dice |
|---|---|---|---|---|
| 1 | 1.1131 | 0.8721 | 0.5030 | 0.3434 |
| 2 | 0.7545 | 0.6893 | 0.5402 | 0.4040 |
| 3 | 0.6494 | 0.6669 | 0.5441 | 0.4361 |
| 4 | 0.6025 | **0.5877** | 0.5853 | 0.4735 |
| 5 | 0.5726 | 0.6073 | 0.5587 | 0.4900 |

Epoch 1 train/val loss: 1.1131 / 0.8721. Epoch 5: 0.5726 / 0.6073. Best epoch (lowest validation loss): 4. Training loss falls every epoch and is still high; validation loss rises slightly at epoch 5 (0.5877 to 0.6073) while foreground Dice keeps improving. One upward tick over five epochs, with a small train-validation gap (+0.035), does not establish overfitting; the model is still learning and mainly underfitting. The empty accuracy at epoch 5 dropped to 0.20, already indicating over-prediction.

### 5.3 Comparing the two models

At epoch 5 the Q8 model was actually ahead of the scratch model on the validation score (0.5587 against 0.5486) and foreground Dice (0.490 against 0.453). The scratch model kept improving to epoch 14 and finished higher (0.6020 against 0.5853), and on the leaderboard and in the analysis below it was stronger on every measure. Several factors differ at once (depth, augmentation, learning rate, input pipeline, number of epochs), so the comparison shows which complete recipe works better, not why.

## 6. Test-Set Behaviour and the Over-Prediction Finding

The first submission (scratch, tau 0.8) predicted an empty mask for only 48.0% of test pairs (a 0.551, b 0.606, c 0.456, d 0.308), whereas the true empty rates in training are about a 0.905, b 0.973, c 0.64, d 0.85. The network flags defects far too often, worst for the rare classes b and d. The leaderboard score (0.4978) was 0.105 below the validation score under reading A. This pointed to the empty-pair part of the metric as the biggest cheap loss, which the next section addresses.

## 7. Post-Processing Analysis

The analysis notebook (`notebooks/milestone-2-analysis.ipynb`) evaluates both models on the same 2,279 validation images, from the same cached pixels, after checking that it reproduces the training-log validation scores (scratch 0.6020, Q8 0.5853).

### 7.1 Methods

- **Threshold (tau):** pixel probability cut-off.
- **Minimum region size:** connected predicted regions (8-connectivity) smaller than N pixels are removed, per class.
- **Confidence gate:** if the most confident pixel of an image-class pair is below G, the whole mask is emptied, per class. It acts as an image-level detector.
- **Flip test-time augmentation (TTA):** average of the plain image and its horizontal and vertical flips.
- **Search:** each predicted region is measured once per candidate threshold (area and true-defect pixels), after which any (N, G) combination is evaluated with array operations only. A coordinate search tunes each class in turn, two passes, with the objective being the average of readings A and B.
- **Honest check:** tuning is repeated on one random half of the images and scored on the other half (both ways). The held-out numbers below are these.

### 7.2 Candidates

| Candidate | raw A | raw B | held-out A | held-out B |
|---|---|---|---|---|
| **scratch + flip TTA** | 0.6119 | 0.4993 | **0.6255** | **0.5836** |
| ensemble (scratch TTA + q8) | 0.6087 | 0.5178 | 0.6220 | 0.5831 |
| ensemble (scratch + q8) | 0.6036 | 0.5095 | 0.6178 | 0.5794 |
| scratch | 0.6020 | 0.4800 | 0.6197 | 0.5712 |
| q8 | 0.5853 | 0.4985 | 0.5905 | 0.5520 |
| q8 + flip TTA | 0.5861 | 0.5044 | 0.5922 | 0.5477 |

("raw" = tau 0.5, nothing else.) Observations:

- Tuning generalises: in-sample tuned B for the best candidate (0.586) is almost identical to its held-out B (0.5836).
- Flip TTA helps the scratch model (trained with flips) but not the Q8 model (never trained on flipped images).
- Adding the Q8 model to an ensemble gives no gain over scratch with TTA.
- Under reading A the gain from tuning is small (+0.014); under B it is large (about +0.08), because it comes almost entirely from fewer false detections.

### 7.3 Selected configuration

`tau = 0.5`, minimum region size `[0, 500, 200, 300]` pixels for classes a to d, confidence gate `[0.98, 0.98, 0.90, 0.98]`, scratch model with flip TTA. After tuning the model predicts a defect on 31% (a), 4% (b), 60% (c) and 26% (d) of validation pairs, against true rates of 9.5%, 2.8%, 36% and 15%: over-prediction is reduced but not eliminated. Three of the gates sit at the top of the search grid (0.98), so a stricter gate may do slightly better.

### 7.4 Where the remaining points are

| Candidate | tuned A | tuned B | score with perfect empty-pair handling | recall of real defects | Dice when found | AUC a / b / c / d |
|---|---|---|---|---|---|---|
| scratch | 0.622 | 0.573 | 0.649 | 0.786 | 0.634 | 0.71 / 0.72 / 0.83 / 0.90 |
| scratch + TTA | 0.626 | 0.586 | 0.648 | 0.770 | 0.646 | 0.72 / 0.73 / 0.84 / 0.91 |
| q8 | 0.593 | 0.554 | 0.614 | 0.752 | 0.598 | 0.69 / 0.47 / 0.81 / 0.89 |
| ensemble | 0.621 | 0.581 | 0.643 | 0.774 | 0.632 | 0.72 / 0.64 / 0.84 / 0.91 |

(AUC = how well the most confident pixel separates real defects from empty pairs; 0.5 is chance.)

- **False-positive control is nearly exhausted.** A perfect handling of empty pairs would give only 0.648. The foreground term is the limit: foreground Dice is about 0.50, from recall 0.77 times Dice 0.65 on found defects.
- **Class-wise:** defect_d is detected best (AUC 0.91), defect_c well (0.84), defects a and b worst (0.72 each). The Q8 model cannot detect defect_b at all (AUC 0.47, it never predicts it), the rarest class with only 313 images.
- **What 0.65 would take:** with empty accuracy between 0.8 and 0.9, foreground Dice must rise from 0.50 to roughly 0.55 to 0.60, which needs better recall and better masks, i.e. a stronger model.

### 7.5 Error analysis

`reports/figures/analysis_cases.png` shows, for the selected candidate, the three most confident false alarms on empty pairs and the three real defects the model is least sure about (image, ground truth, probability map). The numbers above summarise them: roughly a quarter of real defects are missed outright, and the masks of found defects overlap the truth with Dice about 0.65, so both detection (especially for a and b) and boundary quality need improvement.

## 8. Limitations

- The validation set was used both to pick the checkpoint and to tune post-processing. The half-split check shows the tuning itself generalises across validation images, but a small optimism remains.
- The leaderboard metric is only approximately understood (Section 3.3); the expected optimism of reading B is 0.02 to 0.04.
- The scratch-versus-Q8 comparison is confounded (Section 5.3).
- The scratch model was still underfitting; a longer run with a learning-rate schedule would improve it, but was judged a poor use of the limited GPU time (estimated gain of +0.01 to +0.03 against an unreached gap of about 0.09 to the 0.65 cut-off).

## 9. Decisions and Next Steps

1. **Conclude Milestone 2** with the scratch model as the "model built from scratch" for the pass criteria (W&B run `m2-unet-scratch-v1`) and the Q8 run as a second tracked run for comparison.
2. **Milestone 3:** U-Net with a pretrained ResNet encoder (segmentation_models_pytorch) adapted to one grayscale channel; compare BCE, Dice and combined losses and sigmoid versus softmax heads; use flip augmentation, mixed precision and a learning-rate schedule; log every experiment to W&B. The M1 pretrained baseline reached foreground Dice 0.5636 at 8 epochs against about 0.51 for scratch, which is the reason for expecting the largest gain here.
3. **Reuse:** the same holdout split, the analysis notebook (swap the model loader) and the two metric readings, so all models are comparable.
4. **Later milestones:** class balancing for a and b, early stopping, error analysis (M4); tuned thresholds, ensembling and TTA on the best models (M5). The scratch model remains a candidate ensemble member.

## 10. Reproducibility

| Item | Location |
|---|---|
| Scratch training (Dataset, model, loss, W&B) | `notebooks/milestone-2.ipynb` |
| RLE, EDA, Dataset (Q1 to Q7) | `notebooks/milestone-2-questions.ipynb` |
| Q8 baseline training run | `notebooks/milestone-2-q8.ipynb` |
| Two-model analysis and post-processing search | `notebooks/milestone-2-analysis.ipynb` |
| Inference notebook (configurable model, TTA, post-processing) | `notebooks/final_notebook.ipynb` (Kaggle: `DL-22f2001613-notebook-t32026`) |
| Earlier scratch-only inference | `notebooks/inference-m2-unet-scratch.ipynb` |
| Analysis outputs | `reports/analysis_summary.csv`, `reports/post_configs.json`, `reports/figures/` |
| Checkpoints (not in git) | Kaggle outputs of `milestone-2` (`best_unet_scratch.pt`) and `milestone-2-q8` (`best_unet_q8.pt`) |

Seeds are fixed at 42; training used the Kaggle GPU; the split is `data/holdout.csv`.

## 11. Viva Notes (concepts this milestone should make explainable)

- **RLE:** 1-based, column-major; how to encode and decode, why the order matters, why empty masks are NaN in the training file and `" "` in the submission.
- **Dice and the metric:** why overlap, not pixel accuracy, is used; what the three metric components reward; why two readings were compared.
- **Dataset and DataLoader:** one sample versus a batch, the `(channels, height, width)` convention, float32 targets for `BCEWithLogitsLoss`.
- **U-Net:** encoder, decoder, skip connections, why logits are output instead of probabilities.
- **Loss choice:** BCE alone versus BCE plus Dice under 84% empty pairs.
- **Reading curves:** underfitting versus overfitting, and why validation loss can sit below training loss early on.
- **Post-processing:** threshold, minimum region size and confidence gate; why tuning was checked on held-out halves; what AUC of the maximum probability says about how well a gate can work.
- **Honest limits:** what the numbers do and do not show (confounded comparison, approximate leaderboard formula).
