import random
import os
import numpy as np
import torch
from sklearn import metrics


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(targets, outputs, labels):
    report = metrics.classification_report(
        targets, outputs, target_names=labels, digits=4
    )
    print(report)
    return report


def find_optimal_threshold(targets, outputs, num_thresholds=100):
    edges = np.linspace(0, 1, num_thresholds)
    f1_scores = []

    for edge in edges:
        preds = outputs >= edge
        f1 = metrics.f1_score(targets, preds, average="weighted")
        f1_scores.append(f1)

    max_f1 = max(f1_scores)
    optimal_edge = edges[np.argmax(f1_scores)]

    return optimal_edge, max_f1, edges, f1_scores
