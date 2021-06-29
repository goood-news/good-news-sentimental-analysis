from env import conn
import csv

# 한글깨짐 방지
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

curs = conn.cursor()

sql = "select * from CRAWLING2"
curs.execute(sql)
conn.commit()

rows = curs.fetchall()

fp = open('/content/drive/MyDrive/sentimental_analisis/file2.csv', 'w', encoding='utf-8')
myFile = csv.writer(fp)
myFile.writerows(rows)
fp.close()

conn.close()