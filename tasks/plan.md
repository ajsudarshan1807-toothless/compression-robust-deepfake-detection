# Implementation Plan: Dataset Split Fine-Tuning & Training Bug Fixes

## Overview
We will restructure the train/val split logic in `train_clip_final.py` to use a randomized split instead of a sequential one. We will also correct the frequency branch slice index inside the frequency dropout code to preserve all 768 spatial features of the CLIP vision encoder.

## Architecture Decisions
- **Random Permutation before Train/Val Split**: Restructures the sequential split into a global random permutation using `torch.randperm(...)`. This ensures the validation set represents a balanced mixture of both datasets.
- **Slice Correction**: Updates the slice logic to `[:, 768:] = 0` to preserve the first 768 dimensions (CLIP) and drop only the final 128 dimensions (DWT Frequency features).

## Task List

### Phase 1: Preparation & File Mod
- [ ] Task 1: Modify `train_clip_final.py` with corrected index slicing for dropout.
- [ ] Task 2: Modify `train_clip_final.py` with random permutation shuffling for train/val split.

### Checkpoint: Verify Code
- [ ] Code builds without syntax errors.

### Phase 2: Model Training
- [ ] Task 3: Execute `train_clip_final.py` on the CPU to perform feature extraction and head retraining.
- [ ] Task 4: Confirm model saves to `checkpoints/model_h_clip_robust.pt` successfully.

### Phase 3: Verification
- [ ] Task 5: Execute batch evaluation script on the test image set.
- [ ] Task 6: Re-generate visual grid of predictions and verify that the target selfies are correctly classified as Authentic.

## Risks and Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Retraining takes too long on CPU | Medium | Use cached feature extraction shape or reduce epochs if it takes >15 mins (expected: ~5 mins). |
| Overfitting to the training set | Low | Using weight decay (`1e-4`) and dropout (`0.5`) in the fusion head. |
