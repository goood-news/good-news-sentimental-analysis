import pandas as pd

data = pd.read_csv('/content/drive/MyDrive/sentimental_analisis/file.csv')
data.columns = ['CATEGORY', 'PAGE', 'TITLE', 'SOURCE', 'DATE', 'CONTENTS', 'LINK', 'IMAGE', 'FULL_CONTENTS', 'LIKES', 'DISLIKES', 'LABEL']
contents = data['FULL_CONTENTS']

