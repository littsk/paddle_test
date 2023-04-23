#!/bin/bash
set -ex
if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "Directory './logs' created."
else
    echo "Directory './logs' already exists."
fi

seq_lens=(4096 8192 16384 32768)
n_heads=(16 32 64 128)
test_cases=("core_attn_fwd_test"  "core_attn_fwd_bwd_test" "mem_efficient_attn_fwd_test" \
    "mem_efficient_attn_fwd_bwd_test" "flash_attn_fwd_test" "flash_attn_fwd_bwd_test")

for seq_len in "${seq_lens[@]}"
do
    for n_head in "${n_heads[@]}"
    do
        for test_case in "${test_cases[@]}"
        do
            python ./paddle_attn_speed_test.py \
            --yaml ./config/test_cfg.yaml \
            --shape 1,$seq_len,$n_head,64 \
            --test_case $test_case \
            > result_dropout.log
        done
    done
done