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


def evaluation_bleu(golds, predictions):
    bleu_socre = 0
    for key in predictions:
        prediction = predictions[key]
        try:
            gold = golds[key]
        except:
            raise KeyError('{} is not a correct index in e-CARE dataset'.format(key))

        bleu_socre += bleu([gold], prediction)

    avg_bleu = bleu_socre / len(golds)
    return avg_bleu


def evaluation_rouge(golds, predictions):
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    rouge = Rouge()

    for key in predictions:
        prediction = predictions[key]
        try:
            gold = golds[key]
        except:
            raise KeyError('{} is not a correct index in e-CARE dataset'.format(key))

        try:
            scores = rouge.get_scores(prediction, gold)
            rouge_1 += scores[0]['rouge-1']['r']
            rouge_2 += scores[0]['rouge-2']['r']
            rouge_l += scores[0]['rouge-l']['r']
        except:
            continue
    
    avg_rouge1 = rouge_1 / len(golds)
    avg_rouge2 = rouge_2 / len(golds)
    avg_rougel = rouge_l / len(golds)
    return avg_rouge1, avg_rouge2, avg_rougel


def main():
    prediction_file = sys.argv[1]
    gold_path = sys.argv[2]

    prediction_path = os.path.join('prediction_for_eval', prediction_file)
    predictions = json.load(open(prediction_path, 'r'))
    gold_labels = read_gold_data(gold_path)

    avg_bleu = evaluation_bleu(gold_labels, predictions)
    rouge_1, rouge_2, rouge_l = evaluation_rouge(gold_labels, predictions)

    output_file = "_".join(prediction_file.split("_")[:-1] + ["evaluation.json"])
    output_path = os.path.join('evaluation_result', output_file)
    fo = open(output_path, 'w')

    json.dump({"bleu": avg_bleu, "rouge-1": rouge_1, "rouge-2": rouge_2, "rouge-l": rouge_l}, fo)
    print("[Average BLEU]: {}".format(avg_bleu))
    print("[Rouge-1]: {}".format(rouge_1))
    print("[Rouge-2]: {}".format(rouge_2))
    print("[Rouge-l]: {}".format(rouge_l))


if __name__ == '__main__':
    main()