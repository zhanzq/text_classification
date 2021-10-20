#!/usr/bin/env python
# coding: utf-8


#encoding=utf-8
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import pdb

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


import logging
logger = logging.getLogger(__name__)
print([logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR])
logger.parent.setLevel(logging.INFO)
logger.setLevel(logging.INFO)


# 数据切分
data_file = "teaching_comments.txt"


# 加载预训练模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained("/data/zhanzhiqiang/github/teaching_comments/bert-base-chinese")


# 自定义评价标准
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def compute_metrics_test(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=-1)
#     results = metric.compute(predictions=predictions, references=labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }




def preprocess_function(examples):
    # Tokenize the texts
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=30,
        truncation=True,
    )


dataset = load_dataset("csv", encoding="utf-8", data_files="teaching_comments.csv")["train"]
dataset = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )

# shuffle data
total_sz = len(dataset)
idxs = [i for i in range(total_sz)]
random.shuffle(idxs)

# split data
train_sz = int(total_sz*0.7)
valid_sz = int(total_sz*0.2)
train_idxs = idxs[:train_sz]
valid_idxs = idxs[train_sz:train_sz + valid_sz]
test_idxs = idxs[train_sz + valid_sz:]
train_dataset = dataset.select(train_idxs)
eval_dataset = dataset.select(valid_idxs)
test_dataset = dataset.select(test_idxs)


training_args = TrainingArguments(
    output_dir='./output', #存储结果文件的目录
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model = "accuracy", # 最后载入最优模型的评判标准，这里选用acc最高的那个模型参数
    weight_decay=0.01,
    warmup_steps=100,
    evaluation_strategy="steps", #这里设置每100个batch做一次评估，也可以为“epoch”，也就是每个epoch进行一次
    logging_steps = 100,
    seed = 2020,
    do_train = True,
    do_eval = True,
    do_predict = True,
    # max_seq_length = 40,
    # batch_size = 32,
)

# training_args = TrainingArguments(output_dir="./output")
# training_args.task_name = "sst-2"
# training_args.do_train = True
# training_args.max_seq_length = 30
# training_args.batch_size = 64 
# training_args.learning_rate = 2e-5
# training_args.num_train_epochs = 3
# training_args.output_dir = "./output"
# training_args.overwrite_output_dir = True
# 
# # Set seed before initializing model.
# set_seed(training_args.seed)
    

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 3):
    logger.warning(f"Sample {index} of the training set: {train_dataset[index]}.")


# Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
data_collator = default_data_collator

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# Training
if training_args.do_train:
    # checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    # train_result = trainer.train(resume_from_checkpoint=checkpoint)
    train_result = trainer.train()
    # metrics = train_result.metrics
    # metrics["train_samples"] = len(train_dataset)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()



# Evaluation
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    metrics["eval_samples"] = len(eval_dataset)

    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)

# Prediction
if training_args.do_predict:
    test_case_nos = [example["No."] for example in test_dataset]
    test_texts = [example["text"] for example in test_dataset]
    test_labels = [example["label"] for example in test_dataset]
    logger.info("*** Predict ***")
    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")

    metrics["predict_samples"] =  len(test_dataset)

    # trainer.log_metrics("predict", metrics)
    # trainer.save_metrics("predict", metrics)

    predictions = np.argmax(predictions, axis=1)
    test_metrics = compute_metrics_test(labels, predictions)
    print("test result: ", test_metrics)
    output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            writer.write("No.\ttext\tlabel\tprediction\n")
            for index, item in enumerate(predictions):
                # item = label_list[item]
                case_no = test_case_nos[index]
                text = test_texts[index]
                label = test_labels[index]
                writer.write(f"{case_no}\t{text}\t{label}\t{item}\n")
