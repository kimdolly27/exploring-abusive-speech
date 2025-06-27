#!/bin/bash

setups=(P1 P2 P3 P4)
temperatures=(0.00)
mode="base"

for setup in "${setups[@]}"; do
  for temp in "${temperatures[@]}"; do
    echo ">>> Running setup $setup with temperature $temp and mode $mode..."
    python3 prompt-binary-base.py \
      --setup "$setup" \
      --temperature "$temp" \
      --mode "$mode"
  done
done