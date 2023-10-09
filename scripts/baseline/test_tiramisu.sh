bash scripts/train.sh run -y \
    --mode sample200 \
    --tb_logdir .workspace/runs/tiramisu --tiramisu --gpu_model 't4' 2>&1 | tee 1.txt