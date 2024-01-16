Run download_imagenet.sh to prepare directories after downloading Imagenet2012 train and val.

Change /path/to/imagenet accordingly for both train.sh and eval.sh

Run train.sh and change "CUDA_VISIBLE_DEVICES" and the "8" according to gpu circumstances (8 being how many gpus there it runs on).

Change and run train.sh for each model in "vanv2.py", although if it is too much you can run just the first few in order:
- van2_b0
- ablation_van2_nonopw_b0
- ablation_van2_ILKA_b0
- ablation_van2_TLKA_b0
- ablation_van2_nopw_b0
- ablation_van2_scale_b0
- ablation_van2_act_b0
- van2a_b0
- van2_b1
- van2a_b1
- van2_b2
- van2a_b2
- van2_b3
- van2a_b3
- van2_b4
- van2a_b4
- van2_b5
- van2a_b5
- van2_b6
- van2a_b6

Run eval.sh and change checkpoint file accordingly