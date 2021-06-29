# mysql 연결
import pymysql
from sqlalchemy import create_engine

# MySQL Connector using pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

# connection 정보
conn = pymysql.connect(
    host = 'goodnews.cso5uhd7wven.ap-northeast-2.rds.amazonaws.com', # host name
    port = 3306,
    user = 'goodnews', # user name
    password = 'goodnews', # password
    db = 'goodnews', # db name
    charset = 'UTF8'
)





# {} 안에 해당하는 정보 넣기. {}는 지우기.
def get_engine():
    Host = 'goodnews.cso5uhd7wven.ap-northeast-2.rds.amazonaws.com', # host name
    port = 3306,
    User = 'goodnews', # user name
    Password = 'goodnews', # password
    Database = 'goodnews', # db name
    engine = create_engine(f"mysql+mysqldb://{User}:{Password}@{Host}:3306/{Database}", encoding='utf-8')
    return engine

