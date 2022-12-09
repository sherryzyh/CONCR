export TRANSFORMERS_CACHE=./huggingface_cache/

python3 code/train_discriminate.py \
  --data_dir "./data/Causal_Reasoning/" \
  --model_dir "bert-base-cased" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "train.jsonl" \
  --dev "dev.jsonl" \
  --test "test.jsonl" \
  --model_name "bert" \
  --gpu "0" \
  --batch_size 2 \
  --cuda true\
  --epochs 10 \
  --lr 1e-5 \
  --set_seed true \
  --seed 42 \
  --patient 3 \
  --loss_func "BCE" \
  --with_kb true
  --evaluation_strategy "epoch"