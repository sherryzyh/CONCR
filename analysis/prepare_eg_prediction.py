import sys
import os
import json

if __name__ == '__main__':
    input_file = sys.argv[1]
    input_path = os.path.join('raw_output', input_file)
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output = {}
    for i, line in enumerate(lines):
        output[f'dev-{i}'] = line.split(".,")[-1].strip("\n")

    output_file = "_".join(input_file.split("_")[:-2] + ["prediction.json"])
    output_path = os.path.join('prediction_for_eval', output_file)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp)