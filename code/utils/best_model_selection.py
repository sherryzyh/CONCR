import argparse
import os
import json
import jsonlines

def load_metric_log(exp_name):
    with open(os.path.join("/data/output/saved_model", exp_name, "metric_log.json")) as f:
        metric_log = json.load(f)

    return metric_log

def select_epoch(metric_log, best=True, metric="dev_accu", epoch=None,):
    if not best:
        assert isinstance(epoch, int), "Please provide an int epoch number, when not selecting the best epoch"
        assert 0 <= epoch < len(metric_log), "Epoch number is too small or too large"
        return {f'epoch_{epoch}':metric_log[f'epoch_{epoch}']}
    
    if "loss" in metric:
        max_is_better = False
    if "accu" in metric:
        max_is_better = True
    
    best_epoch = "epoch_0"
    best_metric_value = metric_log[best_epoch][metric]

    for epoch, epoch_metrics in metric_log.items():
        if max_is_better:
            if epoch_metrics[metric] > best_metric_value:
                best_epoch = epoch
                best_metric_value = metric_log[best_epoch][metric]
        else:
            if epoch_metrics[metric] < best_metric_value:
                best_epoch = epoch
                best_metric_value = metric_log[best_epoch][metric]

    return best_epoch, metric_log[best_epoch]

def main():
    parser = argparse.ArgumentParser(description='Select Best Model')
    parser.add_argument('--exp_name', '-e',
                        type=str, default=None, help='Experiment Name')
    
    args = parser.parse_args()
    metric_log = load_metric_log(args.exp_name)
    best_epoch, best_epoch_metric = select_epoch(metric_log, metric="dev_accu")
    print(f"Best Epoch {best_epoch}")
    for metric, value in best_epoch_metric.items():
        print(f"{metric:10}: {value}")

if __name__ == '__main__':
    main()