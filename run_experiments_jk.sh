#!/bin/bash

# create folders if missing
[ ! -d "results/" ] && mkdir -p "results/"
[ ! -d "plots/" ] && mkdir -p "plots/"

RESULTS_ROOT="/c/Users/joeyk/MASc/uncertainty_and_calibration_VisionTFormers/results_adversarial_0618/seed0" #"/c/Users/joeyk/MASc/uncertainty_and_calibration_VisionTFormers/results"
SAVE_FILE="./results/results_0707.csv"

run_experiments()
{
  expt_args=(
    --rc_logits_path $1 \
    --df_path $2 \
    --test_dataset $3 \
    --test_logits_path $4 \
    --method "TS" \
    --save_file $SAVE_FILE \
    --seed 0
  )

  python experiments_qc.py "${expt_args[@]}"
  # printf "%s \n" "${expt_args[@]}"
}

# FQ-ViT
settings="vit_small swin_small swin_tiny deit_small deit_tiny"
precisions="w32a32att32 w8a8att32 w4a4att32 w8a4att32 w4a8att32 w6a6att32 w8a8att8 w4a8att8 w8a4att8" # w32a32att32 w8a8att32 w4a4att32"
datasets="ImageNet-A ImageNet-O"
for setting in $settings
do
  for precision in $precisions
  do
    for dataset in $datasets
    do
      lp="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet1k_${precision}_fqvit.p"
      df="${RESULTS_ROOT}/full_fqvit_metrics_2024-07-07.pkl"

      adf=$lp
      if [[ $dataset == 'ImageNet-O' ]]; then
        adf="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet-O_${precision}_fqvit.p ${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet1k-val-for-OOD_${precision}_fqvit.p"
      fi 

      if [[ $dataset == 'ImageNet-A' ]]; then
        adf="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet-A_${precision}_fqvit.p"
      fi 

      run_experiments $lp $df $dataset "${adf[@]}"
    done
  done
done

# RepQ-ViT
precisions="w8a8att32 w4a4att32 w8a4att32 w4a8att32 w6a6att32"
for setting in $settings
do
  for precision in $precisions
  do
    for dataset in $datasets
    do
      lp="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet1k_${precision}_repqvit.p"
      df="${RESULTS_ROOT}/full_repqvit_metrics_2024-07-07.pkl"

      adf=$lp
      if [[ $dataset == 'ImageNet-O' ]]; then
        adf="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet-O_${precision}_repqvit.p ${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet1k-val-for-OOD_${precision}_repqvit.p"
      fi 

      if [[ $dataset == 'ImageNet-A' ]]; then
        adf="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet-A_${precision}_repqvit.p"
      fi 

      run_experiments $lp $df $dataset "${adf[@]}"
    done
  done
done


# IaS-ViT
precisions="w8a8att32 w4a4att32 w8a4att32 w4a8att32 w6a6att32"
for setting in $settings
do
  for precision in $precisions
  do
    for dataset in $datasets
    do
      lp="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet1k_${precision}_iasvit.p"
      df="${RESULTS_ROOT}/full_iasvit_metrics_2024-07-07.pkl"

      adf=$lp
      if [[ $dataset == 'ImageNet-O' ]]; then
        adf="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet-O_${precision}_iasvit.p ${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet1k-val-for-OOD_${precision}_iasvit.p"
      fi 

      if [[ $dataset == 'ImageNet-A' ]]; then
        adf="${RESULTS_ROOT}/logits/${setting}/probs_seed0_${setting}_ImageNet-A_${precision}_iasvit.p"
      fi 

      run_experiments $lp $df $dataset "${adf[@]}"
    done
  done
done
