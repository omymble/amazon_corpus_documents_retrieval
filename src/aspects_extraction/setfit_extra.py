import wandb
from setfit import AbsaModel, AbsaTrainer, TrainingArguments
from config.keys import HF_TOKEN, WANDB_API_KEY
from config.config import DEVICE

from datasets import load_dataset
from transformers import EarlyStoppingCallback
from huggingface_hub import login


def main():
    wandb.login(key=WANDB_API_KEY)
    login(HF_TOKEN)

    model = AbsaModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        spacy_model="en_core_web_lg")
    model.to(DEVICE)

    dataset = load_dataset("omymble/setfit-books-absa")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    args = TrainingArguments(
        output_dir="models/setfit",
        use_amp=True,
        batch_size=256,
        eval_steps=1000,
        save_steps=1000,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="wandb"
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

    trainer.push_to_hub("omymble/setfit-books-absa")
    wandb.finish()


if __name__ == "__main__":
    main()
