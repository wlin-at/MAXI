




import os
from utils.config import get_config
import argparse
import numpy as np
from sklearn import metrics
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    # parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def multi_label_classification_map():
    y_true = np.array([[0, 1, 1, 1], [0, 0, 1, 0], [1, 1, 0, 0]])
    y_scores = np.array([[0.2, 0.6, 0.1, 0.8], [0.4, 0.9, 0.8, 0.6], [0.8, 0.4, 0.5, 0.7]])


    metrics.average_precision_score(y_true, y_scores, average='macro') # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

    threshold = 0.5

    y_pred = []
    for sample in y_scores:
        y_pred.append([1 if i >= threshold else 0 for i in sample])
    y_pred = np.array(y_pred)
