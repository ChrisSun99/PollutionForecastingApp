# -*- coding: utf-8 -*-
"""
Created on 2019/6/7 19:14
@author: luolei

从数据库拉取数据
"""
import sys
import pymysql
import pandas as pd

sys.path.append('../')
pymysql.install_as_MySQLdb()

from mods.config_loader import config


def pull_data_from_db(db_config, start_time, end_time = None):
    """从数据库中拉取原始数据，转为数据表"""
    # Build connection to db.
    db = pymysql.connect(
        host = db_config['host'],            # your host
        user = db_config['user'],            # username
        passwd = db_config['passwd'],        # password
        db = db_config['db'])                # name of the database
    
    # Create a Cursor object to execute queries.
    cursor = db.cursor(pymysql.cursors.DictCursor)
    if end_time is None:
        cursor.execute("""
        SELECT * FROM taiyuan_cityHour WHERE ptime >= {}
        """.format(start_time))
    else:
        cursor.execute("""
        SELECT * FROM taiyuan_cityHour WHERE ptime >= {} AND ptime <= {}
        """.format(start_time, end_time))
    
    # Fetch data.
    result_set = cursor.fetchall()
    cursor.close()
    result = pd.DataFrame(list(result_set))
    
    return result


if __name__ == '__main__':
    # Load db config.
    db_config = config.conf['db_config']
    
    # Result without end time.
    result_without_end = pull_data_from_db(db_config, '2016050603')

    # Result with end time.
    result_with_end = pull_data_from_db(db_config, '2016050603', '2016050723')
