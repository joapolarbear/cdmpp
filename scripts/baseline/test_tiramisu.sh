SAMPLE_NUM=2308
bash scripts/train.sh run -y \
    --mode sample$SAMPLE_NUM \
    --tb_logdir .workspace/runs/tiramisu --tiramisu --gpu_model 't4' 2>&1 | tee 1.txt