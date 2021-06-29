# -*- coding: utf-8 -*-

import pickle as pickle
import json
import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
#  BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BartModel
from transformers import MBartModel, MBartConfig
import transformers
from load_data import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# from torch.optim.lr_scheduler import StepLR

import argparse
from importlib import import_module
from pathlib import Path
import glob
# import wandb
import re
from collections import defaultdict
# from loss import create_criterion
# from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
# from tqdm import tqdm
from tqdm.auto import tqdm
import time
from time import sleep
# from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from transformers import PreTrainedTokenizerFast

import wandb

# seed 고정 
def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def increment_output_dir(output_path, exist_ok=False):
  path = Path(output_path)
  if (path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"


def train(model_dir, args):

  seed_everything(args.seed)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(f"device(GPU) : {torch.cuda.is_available()}")
  num_classes = 2
  
  # load model and tokenizer
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # change model
  if MODEL_NAME == "hyunwoongko/kobart":
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
  if MODEL_NAME == "bert-base-uncased":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  # load dataset
  # datasets_ = load_data("/content/drive/MyDrive/sentimental_analisis/ratings_train.txt")
  # labels_ = datasets_['label'].values
  datasets_ = load_data("/content/drive/MyDrive/sentimental_analisis/file.csv")
  labels_ = datasets_['label'].values
  
# [1 1 0 0 1 0 0 1 1 1] -> 1: 6. 0:4 1.2 0.8
  # train eval split 20% k-fold (5)   # 참고로 저장도 folder 내에 1,2,3,4,5로 되게 하자
  length = len(labels_) # 10
  kf = args.kfold # 1
  class_indexs = defaultdict(list)
  for i, label_ in enumerate(labels_):
    class_indexs[label_].append(i) #  class index [0] = [2,3,5,6], class index[1]=[나머지]
  val_indices = set()
  for index in class_indexs: # stratified: key : 0, 1 classindex[0][0/5:1/5]
    val_indices = (val_indices | set(class_indexs[index][int(len(class_indexs[index])*(kf-1)/5) : int(len(class_indexs[index])*kf/5)]))
  train_indices = set(range(length)) - val_indices

  train_dataset = datasets_.loc[np.array(list(train_indices))]
  train_label = train_dataset['label'].values
  val_dataset = datasets_.loc[np.array(list(val_indices))]
  val_label = val_dataset['label'].values

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_val = tokenized_dataset(val_dataset, tokenizer)

  # make dataset for pytorch.
  train_dataset = CustomDataset(tokenized_train, train_label)
  val_dataset = CustomDataset(tokenized_val, val_label)
  # -- data_loader
  train_loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      num_workers=1,
      shuffle=True,
      pin_memory=torch.cuda.is_available(),
      drop_last=True,
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=args.valid_batch_size,
      num_workers=1,
      shuffle=False,
      pin_memory=torch.cuda.is_available(),
      drop_last=False,
  )

  # setting model hyperparameter
  if args.model_type=='mb':
    config_module = getattr(import_module("transformers"), "MBartConfig")
  elif args.model_type=='BertBase':
    config_module = getattr(import_module("transformers"), "BertConfig")
  else :
    config_module = getattr(import_module("transformers"), args.model_type + "Config")
  
  model_config = config_module.from_pretrained(MODEL_NAME)
  model_config.num_labels = 2
    


#   model_config.hidden_dropout_prob = 0
  # model_config.attention_probs_dropout_prob = 0.5

  # model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  if args.model_type == "Bert":
    model_module = getattr(import_module("model"), "RBERT")
    model = model_module.from_pretrained(MODEL_NAME, config=model_config, args=args)  # args 기존으로 돌리려면 빼줘야 함
  elif args.model_type == "Electra":
    model_module = getattr(import_module("model"), "RElectra")
    model = model_module.from_pretrained(MODEL_NAME, config=model_config, args=args)  # args 기존으로 돌리려면 빼줘야 함
  elif args.model_type == "XLMRoberta":
    model_module = getattr(import_module("model"), "xlmRoBerta")
    model = model_module(config=model_config, args=args)  # args 기존으로 돌리려면 빼줘야 함
  elif args.model_type == "Bart":
    model_module = getattr(import_module("model"), "Bart")
    model = model_module(config=model_config, args=args)  # args 기존으로 돌리려면 빼줘야 함
  elif args.model_type == "mb":
    model_module = getattr(import_module("model"), "mbart")
    model = model_module(config=model_config, args=args)  # args 기존으로 돌리려면 빼줘야 함
    # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-cc25")
  elif args.model_type == "BertBase":
    model_module = getattr(import_module("model"), "BertBaseUncased")
    model = model_module(config=model_config, args=args)

  model.parameters
  model.to(device)
  save_dir = increment_output_dir(os.path.join(model_dir, args.name, str(args.kfold)))

  opt_module = getattr(import_module("transformers"), args.optimizer)  # default: SGD
  optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps = 1e-8
    )
  scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=500, 
    num_training_steps=len(train_loader) * args.epochs, 
    last_epoch=- 1
    )

  # -- logging
  start_time = time.time()
  logger = SummaryWriter(log_dir=save_dir)
  with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(vars(args), f, ensure_ascii=False, indent=4)

  best_val_acc = 0
  best_val_loss = np.inf
  for epoch in range(args.epochs):
    # train loop
    # if epoch == 3:
    #   for name, param in model.named_parameters():
    #     param.requires_grad = True
        # optimizer = opt_module(model.parameters(), lr=1e-5, weight_decay=args.weight_decay, eps = 1e-8)
        # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * (args.epochs-7), last_epoch=- 1)
    model.train()
    loss_value = 0
    matches = 0
    for idx, items in enumerate(train_loader):
      labels = items['labels'].to(device)
      input_ids = items['input_ids'].to(device)
      token_type_ids = items['token_type_ids'].to(device)
      attention_mask = items['attention_mask'].to(device)
      # print(f"attention_mask: ", attention_mask)

      optimizer.zero_grad()
      outs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
      loss = outs[0]
      preds = torch.argmax(outs[1], dim=-1)

      loss.backward()
      optimizer.step()
      scheduler.step()

      loss_value += loss.item()
      matches += (preds == labels).sum().item()
      if (idx + 1) % args.log_interval == 0:
        train_loss = loss_value / args.log_interval
        train_acc = matches / args.batch_size / args.log_interval
        current_lr = get_lr(optimizer)
        print(
            f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
        )
        logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
        logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

        loss_value = 0
        matches = 0

    # val loop
    with torch.no_grad():
      print("Calculating validation results...")
      model.eval()
      val_loss_items = []
      val_acc_items = []
      val_f1_pred = []
      val_f1_y = []
      acc_okay = 0
      count_all = 0
      for idx, items in enumerate(tqdm(val_loader)):
        sleep(0.01)
        labels = items['labels'].to(device)
        input_ids = items['input_ids'].to(device)
        token_type_ids = items['token_type_ids'].to(device)
        attention_mask = items['attention_mask'].to(device)

        outs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)

        preds = torch.argmax(outs[1], dim=-1)
        loss_item = outs[0].item()
        acc_item = (labels == preds).sum().item()

        val_f1_pred += preds.tolist()
        val_f1_y += labels.tolist()

        val_loss_items.append(loss_item)
        val_acc_items.append(acc_item)
        acc_okay += acc_item
        count_all += len(preds)

      cls_report = classification_report(val_f1_y, val_f1_pred, labels=np.arange(42), output_dict=True, zero_division=0)
      val_f1 = np.mean([cls_report[str(i)]['f1-score'] for i in range(num_classes)])


      val_loss = np.sum(val_loss_items) / len(val_loss_items)
      val_acc = acc_okay / count_all
      
      
      if val_acc > best_val_acc:
        print(f"New best model for val acc : {val_acc:4.2%}! saving the best model..")
        model_to_save = model.module if hasattr(model, "module") else model
        # Save 방식이 RBERT는 다른 것 같음 Load도 마찬가지 (args를 저장해줌)
        model_to_save.save_pretrained(f"{save_dir}/best")
        torch.save(args, os.path.join(f"{save_dir}/best", "training_args.bin"))
        # model.save_pretrained(f"{save_dir}/best")
        best_val_acc = val_acc

      if val_loss < best_val_loss:
        best_val_loss = val_loss
      print(
        f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.4}, F1 : {val_f1:4.4} || "
        f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.4}"
      )
      logger.add_scalar("Val/loss", val_loss, epoch)
      logger.add_scalar("Val/accuracy", val_acc, epoch)
      logger.add_scalar("Val/F1(avg)", val_f1, epoch)

      wandb.log({"train_loss": train_loss, "train_acc":train_acc,
          "lr":current_lr, "valid_loss":val_loss, "valid_acc":val_acc})
      s = f'Time elapsed: {(time.time() - start_time)/60: .2f} min'
      print(s)
      print()


if __name__ == '__main__':
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_type', type=str, default='BertBase')
  parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
  
  parser.add_argument('--epochs', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--valid_batch_size', type=int, default=128)
  parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout for fully-connected layers")

  parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')  # 정의가 안됨 모델안에 들어가 있음
  parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
  
  parser.add_argument('--lr', type=float, default=1e-6)
  parser.add_argument('--weight_decay', type=float, default=0.01)
  parser.add_argument('--warmup_steps', type=int, default=500)               # number of warmup steps for learning rate scheduler
  parser.add_argument('--seed' , type=int , default = 42, help='random seed (default: 42)')
  parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
  parser.add_argument('--kfold', type=int, default=1, help='k-fold currunt step number')

  parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
  parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './results'))


  args = parser.parse_args()



  args.epochs = 10
  args.optimizer = 'AdamW'
  args.name = 'new_mbart-large-cc25_kfold_'
  args.pretrained_model = "bert-base-uncased"
  args.model_type = "BertBase"
  args.dropout_rate = 0
  
  # args.pretrained_model = "kykim/bert-kor-base"
  # args.model_type = "Bart"
  # args.pretrained_model = "hyunwoongko/kobart"
  # args.pretrained_model = "monologg/koelectra-base-v3-discriminator"
  # args.pretrained_model = 'monologg/kobert'
  # args.pretrained_model = "google/electra-base-discriminator"
  # args.lr = 5e-6
  # args.output_dir = './results/base_eval20'
  
  model_dir = args.model_dir

  # args.warmup_steps = 300

#   for i in range(4, 6):
  i = 1
  print('='*40)
  print(f"k-fold num : {i}")
  print('='*40)
  args.kfold = i

  wandb.login()
  wandb.init(project='good_news', name=args.name, config=vars(args))

  train(model_dir, args)
