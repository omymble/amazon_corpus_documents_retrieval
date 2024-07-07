# scripts/train_model.py
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_scheduler
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
import os
import time
import pickle
from metrics_logger import MetricsLogger
from config import *


class AspectNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        word_ids = encoding.word_ids()
        label_ids = [-100 if word_id is None else label_encoder.transform([labels[word_id]])[0] for word_id in word_ids]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def train():
    # Load the prepared data
    df = pd.read_pickle(ABSA_PREPROCESSED)

    max_len = 128
    batch_size = 16

    # Encode labels
    global label_encoder
    label_encoder = LabelEncoder()
    all_labels = [label for label_list in df['labels'] for label in label_list]
    label_encoder.fit(all_labels)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = AspectNERDataset(
        texts=df['text'].to_numpy(),
        labels=df['labels'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model.to('cuda')  # Move model to GPU

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0,
                              num_training_steps=len(dataloader) * 3)  # 3 epochs

    log_dir = 'output/fine-tune_AE'
    metrics_logger = MetricsLogger(log_dir)

    # Training loop
    for epoch in range(3):  # You can adjust the number of epochs
        model.train()
        for batch in dataloader:
            start_time = time.time()

            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            end_time = time.time()
            batch_time = end_time - start_time

            # Log metrics
            metrics_logger.log_metrics(train_loss=loss.item(), learning_rate=scheduler.get_last_lr()[0],
                                       batch_time=batch_time)

        # Validation step can be added here if a validation dataset is available

    # Save the model and the label encoder
    model.save_pretrained("models/saved_model")
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    metrics_logger.plot_metrics()


if __name__ == "__main__":
    train()
