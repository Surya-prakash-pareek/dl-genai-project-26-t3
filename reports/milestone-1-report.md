## Milestone 1: Summary of findings and decisions

**Task.** Pixel-level segmentation of 4 surface-defect classes (defect_a to defect_d) in 128x800 grayscale strips; metric = 0.70 foreground Dice + 0.20 false-positive score + 0.10 empty accuracy.

**Findings from the data**
- 11,393 training images (45,572 image-class rows); 3,000 test images. 5,516 images are completely clean.
- Positive images per class: a 1,081, b 313 (rarest), c 4,100 (most frequent), d 1,707. 84.2% of image-class pairs are empty; defect pixels are about 0.77% of all pixels (severe class imbalance).
- 1,198 images contain more than one defect type, but defect masks never share a pixel, so four independent sigmoid outputs are a suitable head.
- Resizing 128x800 to a square would distort the aspect ratio by 6.25x, so images are kept at native size (both dimensions divisible by 32).
- RLE convention: 1-based, column-major, empty mask written as a single space in submissions.

**Decisions**
- Metric: naive mean Dice is misleading here (an all-empty predictor scores 0.84 naive but only 0.30 on the official definition), so the official-style score is used for model selection.
- Split: stratified 80/20 by image over 15 label-combination strata, seed 42 (9,114 train / 2,279 validation); reused in all later milestones.
- Small-region noise filtering analysed; the minimum-area threshold is to be tuned in Milestone 5.

**Baseline (Q12).** U-Net, ResNet-34 encoder with ImageNet weights, 1 input channel, BCE + soft Dice loss, AdamW lr 3e-4, batch 16, 8 epochs, horizontal-flip augmentation. Validation loss fell from 0.6467 to 0.4739 and was still decreasing. Official validation score: 0.6437 / 0.6450 / 0.6455 / 0.6397 at thresholds 0.3 / 0.5 / 0.7 / 0.9; foreground Dice about 0.55 to 0.57. The score is flat across thresholds, so the main limit is segmentation quality, not the threshold. The false-positive pair rate stays high (0.41 to 0.54), which shows the model over-predicts on empty pairs.

**Limitations.** The baseline was not tracked live in W&B (a retrospective summary run was logged) and has not yet been submitted to Kaggle; only the empty dummy submission (0.27337) exists.

**Next.** Milestone 2: reusable Dataset/DataLoader, richer augmentation, W&B tracking, a from-scratch U-Net, and the first real submission.
