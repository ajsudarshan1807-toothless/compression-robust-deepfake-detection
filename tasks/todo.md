# Todo List

## Phase 1: Preparation & File Mod
- [x] Task 1: Modify `train_clip_final.py` with corrected index slicing for dropout.
- [x] Task 2: Modify `train_clip_final.py` with random permutation shuffling for train/val split.

## Checkpoint: Verify Code
- [x] Code builds without syntax errors.

## Phase 2: Model Training
- [x] Task 3: Execute `train_clip_final.py` on the CPU to perform feature extraction and head retraining.
- [x] Task 4: Confirm model saves to `checkpoints/model_h_clip_robust.pt` successfully.

## Phase 3: Verification
- [x] Task 5: Execute batch evaluation script on the test image set.
- [x] Task 6: Re-generate visual grid of predictions and verify that the target selfies are correctly classified as Authentic.
