#!/bin/bash

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
            echo 1,$seq_len,$n_head
        done
    done
done