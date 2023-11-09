# Dilated CNN for audio-to-score alignment

This is an unofficial implementation (WIP!) of the paper ["Structure-Aware Audio-to-Score Alignment using Progressively Dilated Convolutional Neural Networks"](https://arxiv.org/abs/2102.00382).

Instead of training on the MSMD dataset as in the original paper, we train on [ASAP](https://github.com/fosfrancesco/asap-dataset) with synthetic structural augmentations.

## TODO

 - [x] Calculate cross-similarity
 - [x] Add structural augmentations
 - [x] Prepare dataset class
 - [x] Add model implementations
 - [ ] Write training pipeline
 - [ ] Write inference pipeline
 