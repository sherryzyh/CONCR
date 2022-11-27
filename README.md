# Explainable-Causal-Reasoning
 
## Running Causal Reasoning Task
### Parameters
**model_dir**: If use pre-trained model, use the name in Huggingface hub, e.g., "bert-base-cased".
**model_name**: Options include only "bert", "roberta", "albert", "gpt", "gpt2", "bart", or by default "xlnet".

### Baselines
1. fine-tuning with GPT-2
```
python3 code/gpt2_discriminate.py \
  --data_dir "./data/Causal_Reasoning/" \
  --model_dir "gpt2" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "train.jsonl" \
  --dev "dev.jsonl" \
  --model_name "gpt2" \
  --cuda True \
  --gpu "0" \
  --batch_size 64 \
  --epochs 10 \
  --evaluation_step 467 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 10 
```

2. fine-tuning with XLNet-base
```
python3 code/train_discriminate.py \
 --data_dir "./data/Causal_Reasoning/" \
 --model_dir "xlnet-base-cased" \
 --save_dir "./output/saved_model" \
 --log_dir "./output/log" \
 --train "train.jsonl" \
 --dev "dev.jsonl" \
 --model_name "xlnet" \
 --cuda True \
 --gpu "0" \
 --batch_size 64 \
 --epochs 10 \
 --evaluation_step 200 \
 --lr 1e-5 \
 --set_seed True \
 --seed 338 \
 --patient 10 \
 --loss_func "BCE"
 ```
 
 ## Running Explanation Generation Task
 1. fine-tuning with GPT-2
 ```
 python3 code/gpt2_generate.py \
  --data_dir './data/Explanation_Generation/' \
  --model_dir 'gpt2' \
  --save_dir './output/saved_model' \
  --log_dir './output/log' \
  --train 'train.jsonl' \
  --dev 'dev.jsonl' \
  --model_name 'gpt2' \
  --cuda True \
  --gpu '0' \
  --batch_size 32 \
  --epochs 10 \
  --evaluation_step 467 \
  --lr 1e-5 \
  --seed 1024 \
  --patient 10 \
  --length 22
  ```
  
  ## Running Multi-task Training (discriminate-generate)
  ```
  python3 gpt2_multi_task.py \
  --data_dir "./data/" \
  --model_dir "gpt2" \
  --save_dir "./output/saved_model" \
  --log_dir "./output/log" \
  --train "train_full.jsonl" \
  --dev "dev_full.jsonl" \
  --model_name "gpt2" \
  --gpu "0" \
  --batch_size 32 \
  --cuda True\
  --epochs 10 \
  --evaluation_step 933 \
  --lr 1e-5 \
  --set_seed True \
  --seed 338 \
  --patient 10 \
  --length 22 \
  --alpha 0.9 \
  --beam_size 5 \
  --no_repeat_ngram_size 3 \
  --repetition_penalty 1.5 \
  --do_sample True \
  --mode "discriminate_generate"
  ```
