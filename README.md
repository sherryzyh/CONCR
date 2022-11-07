# Explainable-Causal-Reasoning
 
## Running Causal Reasoning Task
1. fine-tuning with GPT-2

```
python3 gpt2_discriminate.py \
  --data_dir "../data/Causal_Reasoning/" \
  --model_dir "gpt2" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "train.jsonl" \
  --dev "dev.jsonl" \
  --model_name "gpt2" \
  --cuda True \
  --gpu "0" \
  --batch_size 64 \
  --epochs 100 \
  --evaluation_step 200 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 3 
```
