# Copyright(c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT License
# [https://opensource.org/license/mit](https://opensource.org/license/mit)
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.



problem_name="GPT_1kctx_10k_lr30e-4"
mycommand=""
runtype="scratch"   # the first run from scratch
#runtype="resume"   # uncomment to resume from the last checkpoint when needed

# Notes: 
# block_size = sequence/context length
# total batch size = gradient_accumulation_steps * batch_size
# if there is a limit on job duration, you will need to implement scratch/resume logic (also check time_limit_seconds and max_iters_per_launch)
# you can adjust max_iters_per_launch to stop training after the specified number of local (within the job) training steps.
# the settings for gradient_accumulation_steps and batch_size are configured for running on 8 nodes (64 GPUs) in parallel.

if [ "$problem_name" = "GPT_1kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=0 --learning_rate=30e-4 --weight_decay=0.1 --warmup_iters=2000 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=1024 --compile=False --batch_size=8 --gradient_accumulation_steps=64 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=14000"
fi

if [ "$problem_name" = "nGPT_1kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=1 --learning_rate=30e-4 --weight_decay=0.0 --warmup_iters=0 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=1024 --compile=False --batch_size=8 --gradient_accumulation_steps=64 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=14000"
fi

if [ "$problem_name" = "GPT_4kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=0 --learning_rate=30e-4 --weight_decay=0.1 --warmup_iters=2000 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=4096 --compile=False --batch_size=2 --gradient_accumulation_steps=256 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=18000"
fi

if [ "$problem_name" = "nGPT_4kctx_10k_lr30e-4" ]; then
    mycommand=" --init_from='$runtype' --use_nGPT=1 --learning_rate=30e-4 --weight_decay=0.0 --warmup_iters=0 --n_layer=24 --n_head=16 --n_embd=1024 --block_size=4096 --compile=False --batch_size=2 --gradient_accumulation_steps=256 --eval_iters=1000 --max_iters=10000 --lr_decay_iters=10000 --time_limit_seconds=103700  --min_lr=0.0 --eval_interval=2000 --max_iters_per_launch=18000"
fi

if [ "$mycommand" != "" ]; then
    torchrun --nnodes 1 --nproc_per_node 8 --rdzv_endpoint=localhost:29501 train.py $mycommand
fi
