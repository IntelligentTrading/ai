import mysql.connector
from mysql.connector import errorcode
import pandas as pd
import numpy as np

def ittconnection(DATABASE='prodcopy'):
    if DATABASE == 'prod':
        config = {
            'user': 'alienbaby',
            'password': 'alienbabymoonangel',
            'host': 'intelligenttrading-aurora-production-primary-cluster.cluster-caexel1tmds5.us-east-1.rds.amazonaws.com',
            'port': '3306',
            'database': 'intelligenttrading_primary',
            'raise_on_warnings': True,
        }
        try:
            db_connection = mysql.connector.connect(**config)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    elif DATABASE == 'stage':
        config = {
            'user': 'alienbaby',
            'password': 'alienbabymoonangel',
            'host': 'intelligenttrading-aurora-production-postgres-cluster.cluster-caexel1tmds5.us-east-1.rds.amazonaws.com',
            'port': '5432',
            'dbname': 'primary_postgres'
        }

        try:
            db_connection = pg.connect(**config)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    elif DATABASE == 'prodcopy':
        config = {
            'user': 'alienbaby',
            'password': 'alienbabymoonangel',
            'host': 'prodclone.caexel1tmds5.us-east-1.rds.amazonaws.com',
            'port': '3306',
            'database': 'intelligenttrading_primary',
            'raise_on_warnings': True,
        }

        try:
            db_connection = mysql.connector.connect(**config)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    return db_connection



def get_raw_price(db_connection, transaction_coin, counter_coin):
    query = "SELECT * FROM indicator_price WHERE transaction_currency='%s' AND counter_currency=%d "
    query = query % (transaction_coin, counter_coin)
    df_sql = pd.read_sql(query, con=db_connection)
    df_sql['timestamp'] = pd.to_datetime(df_sql['timestamp'], unit='s')
    df_sql.index = pd.DatetimeIndex(df_sql.timestamp)

    return df_sql["price"].to_frame()


def get_raw_volume(db_connection, transaction_coin, counter_coin):
    query = "SELECT * FROM indicator_volume WHERE transaction_currency='%s' AND counter_currency=%d "
    query = query % (transaction_coin, counter_coin)
    df_sql = pd.read_sql(query, con=db_connection)
    df_sql['timestamp'] = pd.to_datetime(df_sql['timestamp'], unit='s')
    df_sql.index = pd.DatetimeIndex(df_sql.timestamp)

    return df_sql["volume"].to_frame()


def resample_and_clean(raw_data_frame, resample_pariod='10min'):
    #todo: may be close price, not mean?
    data_ts = raw_data_frame.resample(rule=resample_pariod).mean()
    data_ts['variance'] = raw_data_frame['price'].resample(rule=resample_pariod).var()

    data_ts = data_ts.interpolate()
    #todo: more clever NA handling

    return data_ts

def get_raw_blockchain_data():
    pass