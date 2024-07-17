from config.config import *

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    AdamW,
)

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
import accelerate
from accelerate import Accelerator
from torch.nn.functional import one_hot

import pandas as pd
from sklearn.model_selection import train_test_split


class AspectExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row['sentence_text']
        targets = ', '.join(row['targets'])
        input_text = f"Extract aspect terms from the following input.\ninput: {text}"
        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            targets,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }


def train_t5_base(data, model_name, output_dir):
    data['targets_num'] = data['targets'].apply(lambda x: len(x))
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

    train_df, temp_df = train_test_split(data, test_size=0.3,
                                         stratify=data['targets_num'], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['targets_num'],
                                       random_state=RANDOM_STATE)

    train_dataset = AspectExtractionDataset(train_df, tokenizer)
    val_dataset = AspectExtractionDataset(val_df, tokenizer)
    test_dataset = AspectExtractionDataset(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f'{model_name}_fine-tuned/results'),
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, f'{model_name}_fine-tuned/logs'),
        logging_steps=100,
        save_steps=100,
        evaluation_strategy='steps'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    model.save_pretrained(os.path.join(MODELS, 'saved_models/flan-t5-base_fine-tuned/model'))
    tokenizer.save_pretrained('./models/flan-t5-base_fine-tuned/tokenizer')


def predict_t5_base(model, tokenizer, test_dataset):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            input_ids = test_dataset[i]['input_ids'].unsqueeze(0).to(DEVICE)
            attention_mask = test_dataset[i]['attention_mask'].unsqueeze(0).to(DEVICE)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)
            preds = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            predictions.append(preds.split(', '))
    return predictions



# targets_preprocessed = pd.read_pickle(DATA_PATH + 'targets_preprocessed.pkl')
# targets_preprocessed['targets_num'] = targets_preprocessed['targets'].apply(lambda x: len(x))

# model_name = "google/flan-t5-base"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

# train_df, temp_df = train_test_split(targets_preprocessed, test_size=0.3, stratify=targets_preprocessed['targets_num'], random_state=RANDOM_STATE)
# val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['targets_num'], random_state=RANDOM_STATE)


# train_dataset = AspectExtractionDataset(train_df, tokenizer)
# val_dataset = AspectExtractionDataset(val_df, tokenizer)
# test_dataset = AspectExtractionDataset(test_df, tokenizer)

# training_args = TrainingArguments(
#     output_dir='./models/flan-t5_targets/results',
#     num_train_epochs=4,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./models/flan-t5-targets/logs',
#     logging_steps=100,
#     save_steps=100,
#     evaluation_strategy='steps'
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset
# )