import sys
from logging import getLogger, StreamHandler, Formatter, INFO

import pandas as pd
import numpy as np
import boto3
from awsglue.utils import getResolvedOptions


# ロガーの設定
logger = getLogger('preprocess')
[logger.removeHandler(h) for h in logger.handlers]

logger.setLevel(INFO)
ch = StreamHandler(stream=sys.stdout)
ch.setLevel(INFO)

formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    logger.info('処理を開始')
    
    # コマンドライン引数の確認
    args = getResolvedOptions(sys.argv, ['bucket_name'])
    bucket_name = args['bucket_name']
    logger.info(args)
    logger.info('使用するバケットは s3://{}'.format(bucket_name))
    

    
    # データの読み込み
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(Key='green_tripdata_2019-06.csv',
                         Filename = 'green_tripdata_2019-06.csv')
    
    df = pd.read_csv('green_tripdata_2019-06.csv')
    logger.info('ロードしたデータのサイズは{}'.format(df.shape))
    
    df = df.astype({'VendorID': 'object',
                    'payment_type': 'object',
                    'trip_type': 'object'})
    
    # 使用しないカラムを削除
    drop_columns = ['ehail_fee', 'total_amount',
                    'lpep_dropoff_datetime', 'lpep_pickup_datetime',
                    'PULocationID', 'DOLocationID', 'RatecodeID']
    df = pd.concat([df['total_amount'], df.drop(drop_columns, axis=1)], axis=1)
    
    # カテゴリカルデータのダミー化
    df = pd.get_dummies(df)
    
    # trip_distanceが0の値を抽出、件数確認
    rows_drop = df.index[df["trip_distance"] == 0.00]
    logger.info('移動距離0のデータを削除： {}'.format(df.loc[rows_drop].shape[0]))
    
    # trip_distanceが0の値を削除、件数確認
    df_drop = df.drop(rows_drop)
    logger.info('移動距離0のデータを削除後のサンプルサイズは {}'.format(df_drop.shape[0]))
    
    # データの分割
    train_data, validation_data = np.split(df.sample(frac=1, random_state=1729), [int(0.7 * len(df))])
    logger.info('train_data のサイズは: {}'.format(train_data.shape))
    logger.info('validation_data のサイズは: {}'.format(validation_data.shape))
    
    # データの S3 保存
    train_data.to_csv('train.csv', header=False, index=False)
    validation_data.to_csv('validation.csv', header=False, index=False)
    logger.info('S3 へデータを保存')
    
    
    # バケットへのデータのアップロード
    s3.Object(bucket_name, 'train.csv').upload_file('train.csv')
    s3.Object(bucket_name, 'validation.csv').upload_file('validation.csv')
    logger.info('s3://{} へデータがアップロード'.format(bucket_name))
    
    logger.info('処理を終了')
