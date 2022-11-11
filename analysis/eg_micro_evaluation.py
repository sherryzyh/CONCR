from nltk import bleu
from rouge import Rouge
import jsonlines
import json
import sys
import os


def read_gold_data(path):
    fi = jsonlines.open(path, 'r')
    explanations = {line['index']: line['conceptual_explanation'] for line in fi}
    return explanations


def micro_evaluation_bleu(golds, predictions, output_dict):
    for key in predictions:
        prediction = predictions[key]
        try:
            gold = golds[key]
        except:
            raise KeyError('{} is not a correct index in e-CARE dataset'.format(key))

        bleu_score = bleu([gold], prediction)
        output_dict[key]["bleu"] = bleu_score

    return output_dict


def micro_evaluation_rouge(golds, predictions, output_dict):
    rouge = Rouge()

    for key in predictions:
        prediction = predictions[key]
        try:
            gold = golds[key]
        except:
            raise KeyError('{} is not a correct index in e-CARE dataset'.format(key))

        try:
            scores = rouge.get_scores(prediction, gold)
            rouge_l = scores[0]['rouge-l']['r']
            output_dict[key]["rouge_l"] = rouge_l
        except:
            continue

    return output_dict


def main():
    prediction_file = sys.argv[1]
    gold_path = sys.argv[2]

    prediction_path = os.path.join('prediction_for_eval', prediction_file)
    predictions = json.load(open(prediction_path, 'r'))
    gold_labels = read_gold_data(gold_path)

    output_dict = {k:dict(prediction=v) for k, v in predictions.items()}
    output_dict = micro_evaluation_bleu(gold_labels, predictions, output_dict)
    output_dict = micro_evaluation_rouge(gold_labels, predictions, output_dict)

    output_file = "_".join(prediction_file.split("_")[:-1] + ["micro", "evaluation.json"])
    output_path = os.path.join('evaluation_result', output_file)
    fo = open(output_path, 'w')

    json.dump(output_dict, fo)


if __name__ == '__main__':
    main()





