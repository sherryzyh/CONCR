# CONCR: A CONtrastive learning framework for Causal Reasoning
## 1. Brief Introduction
CONCR ia a CONtrastive learning framework for Causal Reasoning that advances state-of-the-art causal reasoning on the [e-CARE](https://github.com/Waste-Wood/e-CARE) dataset. CONCR is a model-agnostic framework and works better with better sentence encoders. It discards the projection in previous contrastive learning frameworks and uses cosine similarity to score the causal relationship between one premise and one hypothesis. CONCR achieves 77.58% accuracy on BERT-base-uncased and 78.75% on RoBERTa-base, improving previous work by 2.40% and 4.08% respectively.

## 2. Tasks Based on e-CARE Dataset
**Causal Reasoning Task**
Given one premise, denoted as $P$, and two hypotheses candidates, denoted as $H_0$ and $H_1$, this task is formulated as a two-stage task: Firstly, the model takes premise and one hypothesis as the input, and predict its causal score. With these two scores $S_0$ and $S_1$, the predictor select the hypothesis with a higher causal score as the output.


## 3. Experiment Results


## 4. Future Work


## 5. Citations
