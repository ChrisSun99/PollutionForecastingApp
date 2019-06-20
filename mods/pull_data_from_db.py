
import pymysql
import pandas as pd

pymysql.install_as_MySQLdb()


def pull_data_from_db(db_config):
    """从数据库中拉取原始数据，转为数据表"""
    db = pymysql.connect(
        host = db_config['host'],            # your host
        user = db_config['user'],            # username
        passwd = db_config['passwd'],        # password
        db = db_config['db'])                # name of the database
    
    # Create a Cursor object to execute queries.
    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM taiyuan_cityHour")
    
    # cursor.execute("SELECT {}, pm25, {} FROM taiyuan_cityHour".format(feature1, feature2))
    result_set = cursor.fetchall()
    cursor.close()
    result = pd.DataFrame(list(result_set))
    
    return result
