import json
import pickle
import sys
import csv
from sentence_transformers import SentenceTransformer, util

if __name__ == '__main__':
    input_file = sys.argv[1]
    with open('raw_output/' + input_file, 'r') as f:
        lines = f.readlines()
        generated_explanations = [line.strip("\n") for line in lines]
    print("Generated explanations loaded")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    generated_explanation_embeddings = model.encode(generated_explanations)

    output_file = input_file.split(".")[0]
    with open('embeddings/' + output_file + '_embeddings.pkl', "wb") as fOut:
        pickle.dump({output_file: generated_explanations, 'embeddings': generated_explanation_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    print("Generated explanations saved")

    #Load sentences & embeddings from disc
    with open('embeddings/explanation_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        provided_explanation_embeddings = stored_data['embeddings']
    print("Provided explanation embeddings loaded")

    similarities = []
    for i in range(len(provided_explanation_embeddings)):
        cos_sim = util.cos_sim(generated_explanation_embeddings[i], provided_explanation_embeddings[i])
        similarities.append(cos_sim.item())
    print("Explanation embedding similarities generated")
    
    with open('processed_output/' + output_file + '_similarities.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows([[s] for s in similarities])
    print("Explanation embedding similarities saved")