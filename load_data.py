import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def preprocessing_dataset(dataset):
  label = []
  sentence = []
  for i in range(len(dataset)):
    if isinstance(dataset.iloc[i]['FULL_CONTENTS'], str):
      label.append(dataset.iloc[i]['LABEL'])
      st = ''
      for i, s in enumerate(dataset.iloc[i]['FULL_CONTENTS'].split('  ')):
        st += s
        if i == 4:
          break
      sentence.append(st)

  out_dataset = pd.DataFrame({'sentence':sentence, 'label':label})
  return out_dataset


# 파일을 불러옵니다.
def load_data(dataset_dir):
  # load dataset
  dataset = pd.read_csv(dataset_dir, header=None)
  dataset.columns = ['CATEGORY', 'PAGE', 'TITLE', 'SOURCE', 'DATE', 'CONTENTS', 'LINK', 'IMAGE', 'FULL_CONTENTS', 'LIKES', 'DISLIKES', 'LABEL']
  dataset = dataset.loc[dataset['LIKES'] + dataset['DISLIKES'] != 0]
  dataset = preprocessing_dataset(dataset)
  return dataset


def tokenized_dataset(dataset, tokenizer):
  sentence_ent = dataset['sentence'].tolist()

  tokenized_sentences = tokenizer(
      sentence_ent,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=100,
      add_special_tokens=True,
      return_token_type_ids = True
      )

  return tokenized_sentences