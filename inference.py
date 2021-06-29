# 아직 미수정
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from scipy.special import softmax

# import argparse
from importlib import import_module

from env import conn, get_engine

# 한글깨짐 방지
import sys
import io

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device), 
          labels = None, 
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    prob = softmax(logits, axis=-1)[:,1] # axis=0 세로로 합, axis=-1 가로로 합
    # prob = softmax(logits)[:][1]
    # print(prob)
    # print(prob.shape)

    # output_pred += result.tolist()
    output_pred += prob.tolist()
  
  # pred_answer = np.array(output_pred).flatten()
  # verify_df = pd.read_csv("/content/drive/MyDrive/sentimental_analisis/daum_result_today.csv")
  # verify_df['pred'] = pred_answer
  # output.to_csv("/content/drive/MyDrive/sentimental_analisis/good-news-sentimental-analysis/prediction/submission22.csv", index=False)

  return np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer):
  # load dataset
  test_dataset = pd.read_csv(dataset_dir)
  test_dataset.columns = ['CATEGORY', 'PAGE', 'TITLE', 'SOURCE', 'DATE', 'CONTENTS', 'LINK', 'IMAGE', 'FULL_CONTENTS', 'LIKES', 'DISLIKES', 'LABEL']
  test_dataset = preprocessing_dataset(test_dataset)

  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
#   TOK_NAME = "bert-base-multilingual-cased" 
  TOK_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)
  added_token_num = 0
  added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":["[ENT1]", "[/ENT1]", "[ENT2]", "[/ENT2]"]})

  # load my model
  saved_args = torch.load(os.path.join(args.model_dir, "training_args.bin"))
  # saved_args = args

  if args.model_type == "Bert":
    model_module = getattr(import_module("model"), "RBERT")
  elif args.model_type == "Electra":
    model_module = getattr(import_module("model"), "RElectra")
  elif args.model_type == "XLMRoberta":
    model_module = getattr(import_module("model"), "xlmRoBerta")

  model = model_module.from_pretrained(args.model_dir, args=args)  # args 기존으로 돌리려면 빼줘야 함
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "/content/drive/MyDrive/sentimental_analisis/daum_result_today.csv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = CustomDataset(test_dataset, test_label)

  # predict answer
  pred_answer = inference(model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame(pred_answer, columns=['pred'])
  #   output.to_csv('./prediction/submission.csv', index=False)
  output.to_csv(args.outpath, index=False)

  # data to database
  sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
  sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

  curs = conn.cursor()

  df = pd.read_csv(test_dataset_dir)
  df.columns = ['CATEGORY', 'PAGE', 'TITLE', 'SOURCE', 'DATE', 'CONTENTS', 'LINK', 'IMAGE', 'FULL_CONTENTS', 'LIKES', 'DISLIKES', 'LABEL']
  df = df.loc[df['FULL_CONTENTS'].notna()]
  df['PRED'] = pred_answer

  # engine = get_engine()
  # df.to_sql(name='INFERENCE', con=engine, if_exists='append')

  # dataframe filtering

  # similarity check for 0.9 or higher similarity

  for i in range(len(df)):
    if df.iloc[i]['FULL_CONTENTS'] == '':
      continue
    CATEGORY = df.iloc[i]['CATEGORY']
    PAGE = df.iloc[i]['PAGE']
    TITLE = df.iloc[i]['TITLE']
    SOURCE = df.iloc[i]['SOURCE']
    DATE = df.iloc[i]['DATE']
    CONTENTS = df.iloc[i]['CONTENTS']
    LINK = df.iloc[i]['LINK']
    IMAGE = df.iloc[i]['IMAGE']
    FULL_CONTENTS = df.iloc[i]['FULL_CONTENTS']
    LIKES = df.iloc[i]['LIKES']
    DISLIKES = df.iloc[i]['DISLIKES']
    LABEL = df.iloc[i]['LABEL']
    PRED = df.iloc[i]['PRED']

    sql_query = f"INSERT INTO INFERENCE VALUES('{CATEGORY}','{PAGE}','{TITLE}','{SOURCE}','{DATE}','{CONTENTS}','{LINK}','{IMAGE}','{FULL_CONTENTS}','{LIKES}','{DISLIKES}','{LABEL}','{PRED}')"
    curs.execute(sql_query)
    conn.commit()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
 
  # model dir
  parser.add_argument('--model_dir', type=str, default="/opt/ml/code/results/kobert_ent/best")
  parser.add_argument('--outpath', type=str, default="/content/drive/MyDrive/sentimental_analisis/good-news-sentimental-analysis/prediction/submission.csv")
  parser.add_argument('--model_type', type=str, default="Bert")
  parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased")
  parser.add_argument('--dropout_rate', type=float, default=0, help="Dropout for fully-connected layers")
  args = parser.parse_args()

  args.dropout_rate = 0
  # args.pretrained_model = "kykim/bert-kor-base"
  args.model_dir = '/content/drive/MyDrive/sentimental_analisis/good-news-sentimental-analysis/results/new_data_xlm_batch8_kfold_/12/best'
  args.pretrained_model = "xlm-roberta-large"
  args.model_type = "XLMRoberta"
  # args.pretrained_model = "monologg/koelectra-base-v3-discriminator"
  print(args)
  main(args)
