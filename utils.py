import torch
import torch.nn as nn
from transformers import XLMRobertaModel
import numpy as np
import pandas as pd
from konlpy.tag import Okt


class get_similarity(XLMRobertaModel):
  def __init__(self, config, args):
    super(get_similarity, self).__init__(config)
    self.xlmroberta = XLMRobertaModel.from_pretrained("xlm-roberta-large")
    self.num_labels = config.num_labels
  
  def forward(self, input_ids, attention_mask, token_type_ids, labels):
    outputs = self.xlmroberta(
      input_ids, attention_mask=attention_mask
    )
    print("forward outputs: ", outputs)

# class Filter():
#   def __init__(self, full_contents):
#     self.contents = full_contents
#     self.filter_word = ["방화", "살인", "살해", "상해", "폭행", "체포" ,"감금" ,"협박" ,"음모", "강간", "추행", "치상", "강도", "공갈", "묻지마 범죄", "성폭력", "성범죄", "혐오", "학대", "징역", "마약"]

#   def news_filter(self, contents_= self.contents):
#     # print(f"뉴스내용은 {contents_} 입니다")
#     corpus = f"뉴스내용은~~ {contents_} 입니다"
#     return corpus
    