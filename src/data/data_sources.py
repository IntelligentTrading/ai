import os
import mysql.connector
from mysql.connector import errorcode
import psycopg2 as pg
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _ittconnection(DATABASE='prodcopy'):
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
    elif DATABASE == 'postgre_stage':
        config = {
            'user': 'itfcorestage',
            'password': '63YB2P-uZqRpe-NJs6UM_fkG8',
            'host': 'itf-core-aurora-postgresql-stage.caexel1tmds5.us-east-1.rds.amazonaws.com',
            'port': '5432',
            'dbname': 'itf_core_stage_db'
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

    return db_connection


def _get_raw_price(db_connection, transaction_coin, counter_coin):
    query = "SELECT * FROM indicator_price WHERE transaction_currency='%s' AND counter_currency=%d "
    query = query % (transaction_coin, counter_coin)
    df_sql = pd.read_sql(query, con=db_connection)
    df_sql['timestamp'] = pd.to_datetime(df_sql['timestamp'], unit='s')
    df_sql.index = pd.DatetimeIndex(df_sql.timestamp)
    return df_sql["price"].to_frame()


def _get_raw_volume(db_connection, transaction_coin, counter_coin):
    query = "SELECT * FROM indicator_volume WHERE transaction_currency='%s' AND counter_currency=%d "
    query = query % (transaction_coin, counter_coin)
    df_sql = pd.read_sql(query, con=db_connection)
    df_sql['timestamp'] = pd.to_datetime(df_sql['timestamp'], unit='s')
    df_sql.index = pd.DatetimeIndex(df_sql.timestamp)
    return df_sql["volume"].to_frame()


def _get_raw_blockchain_data():
    pass

def get_combined_cleaned_onecoin_df(db_name, transaction_coin, counter_coin, res_period):
    # get raw ts from DB
    logger.info("   retrieve raw coin data:" + transaction_coin + str(counter_coin))

    # form the cache file names
    f_raw_price = "data/raw/" + transaction_coin + str(counter_coin) + "_raw_price.pkl"
    f_raw_volume = "data/raw/" + transaction_coin + str(counter_coin) + "_raw_volume.pkl"

    # read price data from local cache, not from DB if it exists in cache
    # CLEAN cache folder for real run! "data/raw/raw_price.pkl"
    if os.path.isfile(f_raw_price) and os.path.isfile(f_raw_volume):
        raw_price_ts = pd.read_pickle(f_raw_price)
        raw_volume_ts = pd.read_pickle(f_raw_volume)
        logger.info("  ...raw price/volume have been got from the cache...")
    else:
        db_connection = _ittconnection(db_name)
        raw_price_ts = _get_raw_price(db_connection, transaction_coin, counter_coin)
        raw_volume_ts = _get_raw_volume(db_connection, transaction_coin, counter_coin)
        raw_price_ts.to_pickle(f_raw_price)
        raw_volume_ts.to_pickle(f_raw_volume)
        db_connection.close()
        logger.info("  ...raw price/volume have been got from DB...")



    # merge because the timestamps must match, and merge left because price shall have a priority
    raw_data_frame = pd.merge(raw_price_ts, raw_volume_ts, how='left', left_index=True, right_index=True)
    logger.info('> ' + transaction_coin + '/' + str(counter_coin) + ': get raw data from DB, number of time points: ' + str(raw_data_frame.shape[0]))
    raw_data_frame[pd.isnull(raw_data_frame)] = None

    # add variance, resample (for smoothing)
    data_df = raw_data_frame.resample(rule=res_period).mean()  # todo: max?
    data_df['price_var'] = raw_data_frame['price'].resample(rule=res_period).var()
    data_df['volume_var'] = raw_data_frame['volume'].resample(rule=res_period).var()
    data_df = data_df.interpolate()

    del raw_price_ts, raw_volume_ts, raw_data_frame

    return data_df