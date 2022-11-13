import jsonlines
import sys
import pickle
from sentence_transformers import SentenceTransformer, util

if __name__ == '__main__':
    with jsonlines.open('../dev_full.jsonl', 'r') as f:
        hypotheses1 = [line['hypothesis1'] for line in f]
        hypotheses2 = [line['hypothesis2'] for line in f]
        explanations = [line['conceptual_explanation'] for line in f]
        dev_full = [line for line in f]
    print("Dev dataset loaded")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')

    hypothesis1_embeddings = model.encode(hypotheses1)
    hypothesis2_embeddings = model.encode(hypotheses1)
    print("Hypothesis embeddings generated")

    # Store sentences & embeddings on disc
    with open('embeddings/hypothesis1_embeddings.pkl', "wb") as fOut:
        pickle.dump({'hypotheses1': hypotheses1, 'embeddings': hypothesis1_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    with open('embeddings/hypothesis2_embeddings.pkl', "wb") as fOut:
        pickle.dump({'hypotheses2': hypotheses2, 'embeddings': hypothesis2_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Hypothesis embeddings saved")

    explanation_embeddings = model.encode(explanations)
    print("Explanation embeddings generated")

    # Store sentences & embeddings on disc
    with open('embeddings/explanation_embeddings.pkl', "wb") as fOut:
        pickle.dump({'explanations': explanations, 'embeddings': explanation_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Explanation embeddings saved")

    similarities = []
    for i in range(len(hypothesis1_embeddings)):
        cos_sim = util.cos_sim(hypothesis1_embeddings[i], hypothesis2_embeddings[i])
        similarities.append(cos_sim)
    print("Hypothesis embedding similarities generated")

    for i, line in enumerate(dev_full):
        line['hypotheses_cosine_sim'] = cos_sim[i]
    
    with jsonlines.open('../dev_full_augmented.jsonl', 'w') as f:
        f.write_all(dev_full)
    print("Hypothesis embedding similarities saved")