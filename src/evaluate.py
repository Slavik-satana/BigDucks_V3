import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_metrics, find_optimal_threshold


def post_process_and_evaluate(outputs, targets, labels):
    outputs = np.array(outputs)
    targets = np.array(targets)

    # Threshold >= 0.5
    print("Классификации с порогом 0.5:")
    preds = outputs >= 0.5
    calculate_metrics(targets, preds, labels)

    # Подбор оптимального порога
    optimal_edge, max_f1, edges, f1_scores = find_optimal_threshold(targets, outputs)

    # Визуализация F1-score vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(edges, f1_scores, label="F1-score", color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Threshold")
    plt.axvline(
        optimal_edge,
        color="red",
        linestyle="--",
        label=f"Max F1 = {max_f1:.4f} at Threshold = {optimal_edge:.3f}",
    )
    plt.legend()
    plt.grid()
    plt.show()

    # Отчет с оптимальным порогом
    print(f"Оптимальный Threshold: {optimal_edge:.3f} с F1-score: {max_f1:.4f}")
    preds_optimal = outputs >= optimal_edge
    calculate_metrics(targets, preds_optimal, labels)

    return preds_optimal, optimal_edge
