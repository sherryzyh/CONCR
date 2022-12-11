export TRANSFORMERS_CACHE=/data/huggingface_cache/

python3 code/train_discriminate.py \
  --data_dir "./data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --train "train.jsonl" \
  --dev "dev.jsonl" \
  --test "test.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 64 \
  --cuda true\
  --epochs 50 \
  --lr 1e-5 \
  --set_seed true \
  --seed 42 \
  --patient 10 \
  --loss_func "BCE" \
  --with_kb true \
  --evaluation_strategy "epoch"\
  --storage true \
  --storage_dir "/data"