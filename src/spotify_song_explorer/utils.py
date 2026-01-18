import os
import sys  
from src.spotify_song_explorer.exception import CustomException
from src.spotify_song_explorer.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql 

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")




def read_sql_data():
    logging.info("reading sql database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )    
        logging.info("connection established",mydb)
        df=pd.read_sql_query("select * from Spotify_song",mydb)
        print(df.head())

        return df
    


    except Exception as ex:
      raise CustomException(ex)
    