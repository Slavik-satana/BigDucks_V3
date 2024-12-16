import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import *
from utils import seed_everything
from data_processing import (
    load_and_clean_data,
    tokenize_data,
    encode_labels,
    plot_histogram,
    plot_combined_emotions,
    get_alphabet,
)
from dataset import EmotionDataset
from model import EmotionClassifier
from train import train_one_epoch, evaluate
from evaluate import post_process_and_evaluate
import numpy as np


def main():
    # Фиксируем seed
    seed_everything(SEED)

    # Создание директории для логов, если не существует
    os.makedirs(LOG_DIR, exist_ok=True)

    # Загрузка и предобработка данных
    data = load_and_clean_data()

    print("Уникальные символы - Train:", get_alphabet(data["train"]))
    print("Уникальные символы - Valid:", get_alphabet(data["validation"]))

    # Визуализация данных (опционально)
    plot_histogram(data["train"])
    plot_combined_emotions(data["train"])

    # Инициализация токенизатора
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Токенизация данных
    tokenized_data = tokenize_data(data, tokenizer)

    # Кодирование меток
    encoded_data = encode_labels(tokenized_data, LABELS)

    # Создание датасетов
    train_dataset = EmotionDataset(encoded_data["train"], LABELS)
    valid_dataset = EmotionDataset(encoded_data["validation"], LABELS)
    test_dataset = EmotionDataset(encoded_data["test"], LABELS)

    # Создание DataLoader-ов
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Инициализация модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(
        pretrained_model=BASE_MODEL, hidden_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES
    )
    model.to(device)

    # Определение loss-функции и оптимизатора
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )

    # Обучение модели
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        val_outputs, val_targets, val_loss = evaluate(
            model, valid_dataloader, criterion, device
        )

    # Оценка на валидационном наборе и подбор порога
    preds_optimal, optimal_threshold = post_process_and_evaluate(
        val_outputs, val_targets, LABELS
    )

    # Загрузка тестового набора
    tokenized_test = tokenize_data(load_and_clean_data()["test"], tokenizer)
    encoded_test = encode_labels(tokenized_test, LABELS)
    test_dataset = EmotionDataset(encoded_test["test"], LABELS)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Предсказания на тестовом наборе
    test_outputs, _, _ = evaluate(model, test_dataloader, criterion, device)
    test_preds = (np.array(test_outputs) >= optimal_threshold).astype(int)

    # Формирование сабмишена
    df_test = pd.read_csv(TEST_FILE)
    df_submission = pd.DataFrame()
    df_submission["id"] = range(1, len(df_test) + 1)
    df_submission[LABELS] = test_preds
    df_submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"Файл submission.csv - {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
