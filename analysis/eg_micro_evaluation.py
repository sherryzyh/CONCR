from nltk import bleu
from rouge import Rouge
import jsonlines
import json
import sys
import os


def read_gold_data(path):
    fi = jsonlines.open(path, 'r')
    explanations = [line['conceptual_explanation'] for line in fi]
    return explanations


def micro_evaluation_bleu(gold, prediction):
    avg_bleu = bleu([gold], prediction)
    return avg_bleu


def micro_evaluation_rouge(gold, prediction):
    rouge = Rouge()
    scores = rouge.get_scores(prediction, gold)
    rouge1 = scores[0]['rouge-1']['r']
    rouge2 = scores[0]['rouge-2']['r']
    rougel = scores[0]['rouge-l']['r']
    return rouge1, rouge2, rougel


def main():
    prediction_file = sys.argv[1]
    cos_sim_file = prediction_file.split(".")[0] + '_similarities.csv'
    gold_path = '../data/dev_full_augmented.jsonl'

    # load predictions
    with open('raw_output/' + prediction_file, 'r') as f:
        predictions = f.readlines()

    # load gold labels (provided explanations)
    gold_labels = read_gold_data(gold_path)

    # load cosine similarities between generated and provided explanations
    with open('processed_output/' + cos_sim_file, 'r') as f:
        similarities = f.readlines()

    output_dict_list = []
    for i in range(len(gold_labels)):
        avg_bleu = micro_evaluation_bleu(gold_labels[i], predictions[i])
        rouge1, rouge2, rougel = micro_evaluation_rouge(gold_labels[i], predictions[i])
        avg_rouge = sum([rouge1, rouge2, rougel]) / 3
        output_dict_list.append(dict(
            avg_bleu=avg_bleu, 
            rouge1=rouge1, rouge2=rouge2, rougel=rougel, avg_rouge=avg_rouge, 
            cos_sim=float(similarities[i])))

    output_file = "_".join(prediction_file.split("_")[:-1] + ["micro", "EG.json"])
    output_path = os.path.join('micro_evaluation', output_file)
    fo = open(output_path, 'w')
    json.dump(output_dict_list, fo)

if __name__ == '__main__':
    main()





