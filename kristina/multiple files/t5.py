import os
import time
import glob
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# open file in read mode
news = []
header = []
with open('/content/combined.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        if len(row) == 5:
          if row[3] == 'Summary':
            header = row
          if len(row[3]) > 30:
            news.append(row)
    print(len(news))

with open('clean_news.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    for data in news:
      writer.writerow(data)


# delete duplicate
import pandas as pd
import datetime

#reading the csv file
emp = pd.read_csv('/content/combined.csv')
emp.head()
emp.columns

#Dropping duplicates from a particular column
result = emp.drop_duplicates(['URL'],keep='first') #or keep = first or last
print('Result DataFrame:\n', len(result))


print("Starting algorithm")

summaries = list(result["Summary"])
news = list(result["Article"])
domain = list(result["Domain"])
print(len(summaries))
print (len(news))
print(len(domain))


df = pd.DataFrame({'articles' : news, 'summaries' : summaries, 'domain' : domain})
df


#!nvidia-smi
#!pip install --quiet transformers==4.5.0
#!pip install --quiet pytorch-lightining==1.2.7

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap
from transformers import (
   AdamW,
   T5ForConditionalGeneration,
   T5TokenizerFast as T5Tokenizer
)

from tqdm.auto import tqdm