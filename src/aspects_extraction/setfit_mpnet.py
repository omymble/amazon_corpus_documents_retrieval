from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from config.keys import *
from config.config import DEVICE

import wandb
from huggingface_hub import login


def main():
    wandb.login(key=WANDB_API_KEY)
    login(HF_TOKEN)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/paraphrase-mpnet-base-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2",
        spacy_model="en_core_web_lg",
    )
    model.to(DEVICE)
    dataset = load_dataset(SETFIT_DATASET_PATH)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    args = TrainingArguments(
        output_dir="models",
        num_epochs=5,
        use_amp=True,
        batch_size=128,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        load_best_model_at_end=True,
    )

    trainer = AbsaTrainer(
        model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset)
    print(metrics)

    trainer.push_to_hub(SETFIT_MODEL_PATH)
    wandb.finish()


if __name__ == "__main__":
    main()
    wandb.finish()
