import jsonlines
import sys
import os
import csv


def read_gold_data(path):
    fi = jsonlines.open(path, 'r')
    labels = [line['label'] for line in fi]
    return labels


def micro_evaluation_accuracy(gold, prediction):
    if gold == int(prediction):
        return 1
    return 0


def main():
    prediction_file = sys.argv[1]
    gold_path = '../data/dev_full_augmented.jsonl'

    # load predictions
    with open('raw_output/' + prediction_file, 'r') as f:
        predictions = f.readlines()

    # load gold labels
    gold_labels = read_gold_data(gold_path)

    output_list = []
    for i in range(len(gold_labels)):
        accuracy = micro_evaluation_accuracy(gold_labels[i], predictions[i])
        output_list.append(accuracy)

    output_file = "_".join(prediction_file.split("_")[:-1] + ["micro", "CR.csv"])
    output_path = os.path.join('micro_evaluation', output_file)
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows([[a] for a in output_list])


if __name__ == '__main__':
    main()




