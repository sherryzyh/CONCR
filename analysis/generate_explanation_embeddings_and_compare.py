import json
import pickle
from sentence_transformers import SentenceTransformer, util

if __name__ == '__main__':
    eg_explanation_file = 'gpt2_eg_epoch_6_explanations.csv'
    cr_eg_explanation_file = 'gpt2_cr_eg_epoch_8_explanations.csv'
    with open('raw_output/' + eg_explanation_file, 'r') as f:
        lines = f.readlines()
        eg_explanations = [line.strip("\n") for line in lines]
    with open('raw_output/' + cr_eg_explanation_file, 'r') as f:
        lines = f.readlines()
        cr_eg_explanations = [line.strip("\n") for line in lines]
    print("CR and CR-EG explanations loaded")

    eg_explanation_embeddings = model.encode(eg_explanations)
    cr_eg_explanation_embeddings = model.encode(cr_eg_explanations)

    with open('embeddings/eg_explanation_embeddings.pkl', "wb") as fOut:
        pickle.dump({'eg_explanations': eg_explanations, 'embeddings': eg_explanation_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    with open('embeddings/cr_eg_explanation_embeddings.pkl', "wb") as fOut:
        pickle.dump({'cr_eg_explanations': cr_eg_explanations, 'embeddings': cr_eg_explanation_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    