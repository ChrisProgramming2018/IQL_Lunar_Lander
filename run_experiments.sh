#!/bin/bash

declare -i x=0
  
CUDA_VISIBLE_DEVICES=0 python3 main.py --render 1 --wandb 1 --run $x --seed 0 --locexp "13_11_run_0" &

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=1 python3 main.py --render 1 --wandb 1 --run $x --seed 0 --locexp "13_11_run_1" &

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=2 python3 main.py --render 1 --wandb 1 --run $x --seed 1 --locexp "13_11_run_2" &

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=3 python3 main.py --render 1 --wandb 1 --run $x --seed 2 --locexp "13_11_run_3" &

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=4 python3 main.py --render 1 --wandb 1 --run $x --seed 3 --locexp "13_11_run_4" &

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=5 python3 main.py --render 1 --wandb 1 --run $x --seed 4 --locexp "13_11_run_5" & 

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=6 python3 main.py --render 1 --wandb 1 --run $x --seed 5 --locexp "13_11_run_6" &

x=x+1
   
sleep 5

CUDA_VISIBLE_DEVICES=7 python3 main.py --render 1 --wandb 1 --run $x --seed 6 --locexp "13_11_run_7" &

x=x+1
   
sleep 5
