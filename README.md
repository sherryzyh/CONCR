# CONCR: A CONtrastive learning framework for Causal Reasoning
## 1. Brief Introduction
CONCR ia a CONtrastive learning framework for Causal Reasoning that advances state-of-the-art causal reasoning on the [e-CARE](https://github.com/Waste-Wood/e-CARE) dataset. CONCR is a model-agnostic framework for contrastively learn the causality-embedded representation. It encodes the sentence separately unlike the previous two-sentence encoding in e-CARE. With the sentence representation, CONCR discards the projection which is widely used in the contrastive learning but use a simple cosine simlarity scorer to calculate the causal score between given premise-hypothesis pair. In the training, the positive samples are constructed by pairing the premise with its correct hypothesis and the negative samples are contructed by pairing premise with any other hypothesis within the same mini-batch. A contrastive cross-entropy learning objective is used to enforce the model to learn the causality-embedded representation. CONCR achieves 77.58% accuracy on BERT-base-uncased and 78.75% on RoBERTa-base, improving previous work by 2.40% and 4.08% respectively.

## 2. Tasks Based on e-CARE Dataset
**Causal Reasoning Task**

Given one premise, denoted as $P$, and two hypotheses candidates, denoted as $H_0$ and $H_1$, this task is formulated as a two-stage task: Firstly, the model takes premise and one hypothesis as the input, and predict its causal score. With these two scores $S_0$ and $S_1$, the predictor select the hypothesis with a higher causal score as the output.

**Explanation Generation Task**

Given one premise $P$ and the correct hypothesis $H$, this task is asking the model to take $P$ and $H$ as the input and generate a free-text-formed explanation $E$ for this cause-effect pair.

## 3. Experiment Results
|Language Model|Accuracy|
| |e-Care|CONCR|

## 4. Future Work
We have three potential future directions. Firstly, we can evaluate this framework on other causal reasoning tasks like COPA.

Moreover, currently there is no appropriate metric for evaluating explanations. Therefore, designing a reasonable metric that can be used to measure the quality of the generated explanations in causal reasoning can be another future work.

In addition, while knowledge bases have the potential to provide the model with important domain knowledge, we have yet to find an effective method to leverage knowledge bases for causal reasoning. Future work can consider more advanced designs with the goal to find the relevant knowledge and inject it in a way that helps with causal reasoning.
