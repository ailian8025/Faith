"""
https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb
"""

import os
from functools import partial

import wandb

os.environ["http_proxy"] = "http://localhost:1080"
os.environ["https_proxy"] = "http://localhost:1080"

if __name__ == '__main__':
    wandb.init(project="Faith", entity="ailian")
    proxies = {"http": "http://localhost:1080", "https": "http://localhost:1080"}
    from datasets import load_dataset

    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

    model_checkpoint = "gpt2"
    tokenizer_checkpoint = "sgugger/gpt2-like-tokenizer"

    from transformers import AutoTokenizer
    # PS: it really make me feel confuse, it must set the proxies here !!
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, proxies=proxies)


    def tokenize_function(tokenizer, examples):
        return tokenizer(examples["text"])


    tokenize_function2 = partial(tokenize_function, tokenizer)

    tokenized_datasets = datasets.map(tokenize_function2, batched=True, num_proc=4, remove_columns=["text"])

    print(tokenized_datasets["train"][1])


    def group_texts(examples):
        block_size = 128
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    tokenizer.decode(lm_datasets["train"][1]["input_ids"])

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_config(config)

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        f"{model_checkpoint}-wikitext2",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )
    import math

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
