import itertools
import re
from collections import defaultdict

from datasets import load_dataset
from config import TRAIN_FILE, VALID_FILE, TEST_FILE, LABELS, MAX_LEN
import matplotlib.pyplot as plt


def load_and_clean_data():
    data = load_dataset(
        "csv",
        data_files={"train": TRAIN_FILE, "validation": VALID_FILE, "test": TEST_FILE},
    )

    # Очистка текста
    def cleaner(example):
        example["text"] = example["text"].lower()
        example["text"] = re.sub(r"[^a-zа-я\d]", " ", example["text"])
        example["text"] = re.sub(r"\s+", " ", example["text"])
        example["text"] = example["text"].strip()
        return example

    data = data.map(cleaner)
    return data


def tokenize_data(data, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=True,
            padding="max_length",
        )

    tokenized_data = data.map(tokenize_function, batched=True)
    return tokenized_data


def encode_labels(data, labels):
    def one_hot_to_list(example):
        emotions = [example[emotion] for emotion in labels]
        example["one_hot_labels"] = emotions
        return example

    data = data.map(one_hot_to_list)
    return data


def get_alphabet(data_split):
    alphabet = set()
    for sample in data_split:
        uniq_chars = set(sample["text"])
        alphabet.update(uniq_chars)
    return alphabet


def plot_histogram(data_split):
    class_counts = {emotion: 0 for emotion in LABELS}

    for sample in data_split:
        for emotion in LABELS:
            class_counts[emotion] += sample[emotion]

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel("Классы эмоций")
    plt.ylabel("Количество")
    plt.title("Распределение по классам эмоций")
    plt.xticks(rotation=45)
    plt.show()


def plot_combined_emotions(data_split):
    emotion_pairs = defaultdict(int)

    for entry in data_split:
        emotions = {key: value for key, value in entry.items() if key in LABELS}
        active_emotions = [emotion for emotion, value in emotions.items() if value > 0]

        for pair in itertools.combinations(sorted(active_emotions), 2):
            emotion_pairs[pair] += 1

    sorted_pairs = sorted(emotion_pairs.items(), key=lambda x: x[1], reverse=True)
    pairs = [f"{pair[0]} & {pair[1]}" for pair, _ in sorted_pairs]
    counts = [count for _, count in sorted_pairs]

    plt.figure(figsize=(12, 6))
    plt.bar(pairs, counts)
    plt.xlabel("Пары эмоций")
    plt.ylabel("Количество")
    plt.title("Совместное появление эмоций")
    plt.xticks(rotation=45)
    plt.show()
