# SurfaceGuard AI: Industrial Surface Anomaly Segmentation

**Final Project Report**

Surya Prakash Pareek, roll number 22f2001613 Deep Learning and Generative AI Project (BSDA2001P), BS in Data Science and Applications, IIT Madras. Term T3-2026.

**Final public leaderboard score: \[TO FILL: final score\], reached on \[TO FILL: date\].**

## 1. Problem statement

The task is pixel-level segmentation of four types of surface defects (defect\_a to defect\_d) in images of steel strips, set as a Kaggle competition. For every image and every defect type the system must output a binary mask marking exactly the pixels that belong to that defect, or an empty mask when the defect is absent. Defects differ widely in size, shape, contrast and number of occurrences, and one image can contain several defect types.

**Input and output.** The input is one 8-bit grayscale image of 128 x 800 pixels (a line-scan strip). The output is four masks of the same size, one per defect type, submitted in run-length encoding (RLE).

**Data.** The training set has 11,393 images with annotations stored in a long table of 45,572 rows (11,393 images times 4 defect types). The test set has 3,000 images, so the submission has 12,000 rows, one per image and defect type. RLE stores a mask as pairs of start position and run length, with 1-based positions counted down the columns first (column-major order). An empty mask is a missing value in the training table and a single space in the submission file.

**Metric.** Each submission is scored by

Score = 0.70 x Dice\_fg + 0.20 x S\_fp + 0.10 x A\_empty

- Dice\_fg is the mean Dice coefficient, 2|X and Y| / (|X| + |Y|), between the predicted pixel set X and the true pixel set Y over the image and defect pairs whose true mask is not empty. It rewards accurate localisation and carries most of the weight.
- S\_fp penalises predicted pixels on pairs whose true mask is empty, so a model that reports defects that are not there loses points.
- A\_empty rewards correctly predicting an empty mask when a defect type is absent.

The exact computation of the last two parts is not published. Two working readings were therefore used throughout: reading A, in which S\_fp is one minus the mean predicted fraction of the canvas on empty pairs, and reading B, a stricter per-pair reading with score 0.7 x Dice\_fg + 0.3 x A\_empty in which any predicted pixel on an empty pair is a failure. Reading B tracked the leaderboard much more closely and was used to tune the final system.

**Result.** The final system is a pretrained ResNet-34 U-Net with a classification head, flip test-time augmentation and per-class post-processing, with a public leaderboard score of \[TO FILL: final score\] against 0.65 for the course cut-off.

## 2. Data understanding and preparation

Exploration of the training data (Milestone 1 and the Milestone 2 questions) gave the following facts.

| Fact | Value |
| --- | --- |
| Training and test images | 11,393 and 3,000 |
| Image and defect-type pairs in training | 45,572, of which 84.2% are empty |
| Share of all pixels that are defect | about 0.77% |
| Images with no defect at all | 5,516 (48.4%) |
| Images containing defect\_a | 1,081 (9.5%) |
| Images containing defect\_b (rarest) | 313 (2.7%) |
| Images containing defect\_c (most common) | 4,100 (36.0%) |
| Images containing defect\_d | 1,707 (15.0%) |
| Images with more than one defect type | 1,198 |
| Pixels shared by two defect types | none: the masks of different types never overlap |

The strong class imbalance and the large share of empty pairs shaped every later decision: the loss, the output head and the post-processing.

**Decoding and checking RLE.** A decoder fills a flat array from the start and length pairs, converts to 0-based indices, clips runs to the 102,400-pixel canvas and reshapes in column-major order. It was checked with a hand-worked toy mask, with the rule that the sum of run lengths equals the mask pixel count, and with a round trip (encode, then decode) on 200 random 128 x 800 masks. An empty entry decodes to an all-zero mask.

**Input decisions.** The network input has one channel because the images are grayscale. The native 128 x 800 size is kept: resizing to a square would distort the aspect ratio by a factor of 6.25, and both sides are divisible by 32, so networks with five downsampling stages accept the image unchanged. Pixels are divided by 255 and then normalised with the training-set mean 0.3314 and standard deviation 0.1892, estimated on 300 training images.

**Train and validation split.** The split is by image, so the four rows of one image never fall on both sides, and it is stratified over the 15 label combinations that occur (which defect types an image contains), with ratio 80/20 and seed 42. This gives 9,114 training and 2,279 validation images. Stratification keeps the rare defect\_b in both parts. The same split was used for every model, so all validation scores are comparable.

## 3. Pipeline overview

Figure 1 shows the stages from the raw tables to the submission file. Training and inference are separate: training produces checkpoints, and inference loads them to predict the test set.

&#91;TO INSERT: Figure 1, pipeline diagram\]

The code lives in notebooks, one set per milestone, in the project repository (GitHub, branches milestone-1 to milestone-5). Training notebooks cover the scratch model (Milestone 2), the pretrained model (Milestone 3) and the longer and improved runs (Milestone 4). A separate analysis notebook tunes the post-processing on the validation set, and the inference notebook (Kaggle notebook DL-22f2001613-notebook-t32026) writes the submission file.

## 4. Model architectures

Three models were built, as required: one from scratch, one pretrained, and one of my own choice.

|  | Model 1: from scratch | Model 2: pretrained | Model 3: own choice |
| --- | --- | --- | --- |
| Architecture | U-Net, 4 encoder levels (32, 64, 128, 256 channels) and a 512-channel bottleneck | U-Net with ResNet-34 encoder and 4 skip connections, plus an image-level classification head | Model 2 plus a full-resolution skip branch |
| Encoder | double 3 x 3 convolutions with batch normalisation, max-pooling | ResNet-34 (21.3 million parameters) | ResNet-34 |
| Starting weights | random | ImageNet, first convolution adapted to one channel by summing the three RGB filters | as Model 2 |
| Decoder | transposed convolutions and skip connections | 4 upsampling blocks (256, 128, 64, 32 channels) with skip connections | as Model 2, plus a small convolution branch on the raw image concatenated before the last block |
| Parameters | 7,762,564 | 24,432,072 | 24,436,888 |
| Why chosen | a baseline to build and validate the whole pipeline without any dependency | pretrained filters for edges and textures suit small, imbalanced defect data | the ResNet stem halves the resolution at once, so thin defects such as hairline cracks lose detail; the extra branch gives the head full-resolution information |

Every model ends in a 1 x 1 convolution that produces four logits per pixel, one channel per defect type. A sigmoid turns each logit into a probability between 0 and 1, independently per type, which also allows several defect types in one image. Probabilities above a threshold become the binary mask. The classification head of Models 2 and 3 (global average pooling, dropout and a linear layer) gives four image-level presence probabilities, which are used in post-processing to decide whether a defect type is present in an image.

## 5. Training details

| Setting | Model 1 | Model 2 | Model 3 |
| --- | --- | --- | --- |
| Loss | BCE with logits plus soft Dice on pairs that contain a defect | same, plus 0.5 times the BCE of the classification head | same as Model 2 |
| Optimiser | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 |
| Learning rate | 1e-3, constant | decoder 1e-3, encoder 3e-4, one epoch of warm-up then cosine decay to 2% | as Model 2 |
| Batch size and epochs | 16, 15 | 16, 20 | 16, 30 |
| Weight averaging | none | none | exponential moving average (decay 0.999) |
| Augmentation | horizontal and vertical flips, brightness shift +/-0.1, contrast factor 0.8 to 1.2 | same | same, plus random crops of width 512 and oversampling of images with rare defects |
| Precision and clipping | float32 | mixed precision (AMP), gradient clipping at norm 5 | as Model 2 |
| Hardware and time | Kaggle GPU, about 372 s per epoch | Kaggle T4 GPU, about 125 s per epoch | Kaggle T4 GPU |
| Seed | 42 | 42 | 42 |

**Reasons.** Binary cross-entropy alone is minimised by predicting no defect anywhere, because 84% of pairs are empty; the soft Dice term rewards overlap with real defects and the classification loss trains the image-level detector. Flips are valid because defects have no preferred position or orientation on the strip, and brightness and contrast changes imitate lighting variation. Pretrained encoders use a lower learning rate than the randomly initialised decoder because they only need adjusting.

**Model selection.** Checkpoints were chosen on the validation set: for Model 1 the best score at threshold 0.5 (epoch 14 of 15), for Model 2 the best average of readings A and B at threshold 0.5 (epoch 19 of 20). Final thresholds were tuned separately (Section 7).

**Experiment tracking.** All runs were logged to Weights and Biases (project 22f2001613-t32026) with their configuration, per-epoch train and validation loss, readings A and B, foreground Dice, empty accuracy, classification AUC, learning rate and epoch time, plus the loss curves. Key runs: m2-unet-scratch-v1, m3-unet-resnet34-v1, m4-r34-cos40 and m4-r34fr-aug.

## 6. Experiments and results

Each row changes one thing from the row it builds on. Validation scores are at threshold 0.5 unless stated; "held-out" means tuned on one half of the validation images and scored on the other half.

| No. | Experiment | Validation (A / B) | Leaderboard | What it showed |
| --- | --- | --- | --- | --- |
| 1 | Pretrained ResNet-34 U-Net, 8 epochs (Milestone 1 baseline) | A 0.645 at threshold 0.7, foreground Dice 0.564 | not submitted | Pretrained encoders are far ahead of random initialisation; this run was not logged to W&B and was repeated as experiment 5 |
| 2 | Scratch U-Net, 15 epochs | 0.602 / 0.480 | 0.4978 (threshold 0.8) | Both losses still falling; defects were reported on 45%, 39%, 54% and 69% of test pairs (a to d) against true rates of 9.5%, 2.7%, 36% and 15% |
| 3 | Small 3-stage U-Net, 3 channels, no augmentation, 5 epochs | 0.585 / 0.499 | 0.45898 | Weaker than experiment 2; it never detects defect\_b (AUC 0.47) |
| 4 | Scratch U-Net plus flip TTA and tuned post-processing | held-out 0.626 / 0.584 | 0.56021 | Post-processing alone added about 0.06 on the leaderboard |
| 5 | ResNet-34 U-Net, 20 epochs, classification head, AMP, cosine schedule | 0.719 / 0.664 | not submitted raw | Foreground Dice rose from 0.52 to 0.64 and the classification AUC to 0.93 to 0.99 |
| 6 | Experiment 5 plus flip TTA and tuned classification gate and area filter | held-out 0.731 / 0.719 | 0.65173 | First score above the 0.65 cut-off |
| 7 | ResNet-34 U-Net, 40 epochs with EMA weights (best epoch 35) | 0.722 / 0.678 (gated B 0.697) | 0.64441 with an untuned gate of 0.5 | Validation loss was lowest at epoch 21 (0.456) and then rose to 0.475 while training loss kept falling to 0.307, i.e. overfitting; the untuned gate cost accuracy |
| 8 | Post-processing re-tuned for the best checkpoints (analysis notebook) | \[TO FILL: held-out A / B\] | \[TO FILL\] | \[TO FILL\] |
| 9 | Model 3: full-resolution skip, crop augmentation, oversampling, 30 epochs | \[TO FILL: A / B\] | \[TO FILL\] | \[TO FILL\] |

**The three required models on the same validation set (threshold 0.5, no post-processing).**

| Model | Parameters | Score A | Score B | Foreground Dice |
| --- | --- | --- | --- | --- |
| 1: scratch U-Net (epoch 14) | 7.76 million | 0.602 | 0.480 | 0.523 |
| 2: pretrained ResNet-34 U-Net (epoch 19) | 24.43 million | 0.719 | 0.664 | 0.638 |
| 3: \[TO FILL: model name\] | 24.44 million | \[TO FILL\] | \[TO FILL\] | \[TO FILL\] |

&#91;TO INSERT: Figure 2, leaderboard score over submissions with the 0.65 cut-off line\]

**What helped most.** The pretrained encoder gave the largest single gain (validation score A from 0.602 to 0.719), because its filters already describe edges and textures and the classification head separates defective from clean images far better. Per-class post-processing was second: it moved the scratch model from 0.4978 to 0.5602 and the pretrained model above the cut-off, because the raw models report many false defects on empty pairs. Longer training did not help: 40 epochs gave about 0.012 more than 20 on the gated validation score and began to overfit.

## 7. Post-processing and inference

**Thresholds.** A shared pixel threshold was searched among 0.5 and 0.7 and 0.5 was selected. Per class, a minimum region area and a confidence gate were tuned by coordinate search over a grid of 10 area values and 11 gate values, using reading B as the objective. The gate is applied to the image-level presence probability from the classification head; for models without that head, the highest pixel probability is used. The configuration of the 0.65173 submission was minimum areas of 100, 50, 200 and 200 pixels and gates of 0.1, 0.2, 0.2 and 0.2 for defect\_a to defect\_d. The final configuration is \[TO FILL: final configuration\].

**Area filter and gate.** Connected regions (8-connectivity) smaller than the class minimum are removed as noise, and a defect type is dropped from an image when its presence probability is below the gate. Tuning raised validation reading B for the pretrained model from 0.664 to 0.711 in-sample and 0.708 on held-out halves, so the gain is not an artefact of fitting the validation set.

**Test-time augmentation and ensembling.** Predictions are averaged over the original image and its horizontal and vertical flips. This added about 0.01 on validation for the scratch model (A 0.602 to 0.612) and about 0.011 for the pretrained model (held-out B 0.708 to 0.719). Averaging the scratch and pretrained models did not help (held-out B 0.716 and 0.693 for weights 2:1 and 1:1, against 0.719 for the pretrained model alone), because the scratch model is too weak. \[TO FILL: result of the final ensemble, if used.\]

**Building and checking the submission.** Probabilities are post-processed per image and defect type, encoded to RLE (1-based, column-major, a single space for an empty mask) and written in the row order of the sample submission. Checks: exactly 12,000 rows, the same columns as the sample file, and the share of empty predictions per class compared with the expected rates.

## 8. Error analysis

**Main failures and fixes.**

- Over-prediction. The raw scratch model reported defects on far more pairs than occur. Cause: 84% empty pairs and a mask loss that does not penalise clean-image noise strongly. Fix: classification gate, area filter and threshold, which raised the leaderboard score by about 0.06.
- Missing class. The small 3-stage model never predicted defect\_b (AUC 0.47). Cause: the rarest class (313 images), no augmentation and only 5 epochs. Fix: abandoned in favour of the pretrained model, whose classification AUC for defect\_b is 0.965.
- Overfitting. The 40-epoch run reached its lowest validation loss at epoch 21 and then worsened. Fix: no further lengthening, EMA weights, and stronger augmentation in Model 3.
- Untuned post-processing. Submitting with a gate of 0.5 for all classes scored 0.64441, below the tuned 0.65173. Fix: re-tune the gates for every checkpoint before submitting.

**Per-class results** (pretrained ResNet-34 U-Net, validation set, threshold 0.5, before post-processing; the last column is after tuning and TTA).

| Defect | True rate | Predicted rate | Recall of real defects | Dice on positive pairs | Empty accuracy | Classification AUC | Predicted rate after tuning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a | 9.5% | 36.0% | 74.1% | 0.447 | 0.692 | 0.929 | 14.2% |
| b | 2.8% | 32.1% | 82.5% | 0.458 | 0.697 | 0.965 | 3.6% |
| c | 36.0% | 62.0% | 94.1% | 0.655 | 0.574 | 0.958 | 39.7% |
| d | 15.0% | 22.1% | 94.4% | 0.752 | 0.908 | 0.987 | 15.8% |

Defects a and b remain the hardest: their masks are the thinnest and rarest, with Dice about 0.45 against 0.66 for c and 0.75 for d, and defect\_a still misses a quarter of its real defects. After tuning, false-defect control is nearly exhausted: a perfect decision on every empty pair would raise the validation score only from 0.721 to 0.737. The remaining loss is in recall (0.85) and mask quality (Dice 0.73 on defects that are found).

**Validation versus leaderboard.** The leaderboard score is consistently lower than the held-out validation score: 0.584 against 0.560 for the scratch model and 0.719 against 0.652 for the pretrained model. Likely reasons: the leaderboard formula differs from both working readings; the test set may contain a different mix of defects than the validation split; and the validation split was used both to select checkpoints and to tune post-processing, which makes validation scores slightly optimistic.

&#91;TO INSERT: Figure 3, example validation images with the true mask beside the predicted probability map, for two false alarms and two missed defects\]

## 9. Insights and next steps

**Main lessons.**

1. A pretrained encoder was the single largest improvement on this small, imbalanced dataset.
2. On a metric that rewards empty predictions, an image-level presence head and per-class tuning of the gate and area filter are worth as much as model changes.
3. Validation scores overestimate leaderboard scores; decisions were safer when based on held-out halves of the validation set and on a metric reading that tracked the leaderboard.
4. Training for longer is not a substitute for regularisation: the 40-epoch run overfitted after about 21 epochs.
5. Measure before scaling: every change was tracked in W&B so that its effect, and its failure, could be explained.

**Next steps.**

- Oversample or weight defects a and b more strongly, because they have the lowest recall and Dice.
- Use a decoder with full-resolution detail and dilated convolutions for thin defects, to raise mask quality.
- Add stronger augmentation (random crops, scale and noise) before training longer.
- Ensemble diverse pretrained encoders (for example ResNeXt-50 and EfficientNet) with the gate tuned on the ensemble.
- Retrain the best configuration on training plus validation images with the tuned post-processing fixed.

The work stopped at this point because of the final submission deadline and the limited GPU time available; the remaining gains require new training runs rather than further tuning.
