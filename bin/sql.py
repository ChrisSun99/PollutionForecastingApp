
import pymysql
import pandas as pd
import sys

sys.path.append('../')

from mods.config_loader import config

pymysql.install_as_MySQLdb()

db = pymysql.connect(
    host=config.conf['pymysql']['host'],            # your host
    user=config.conf['pymysql']['user'],            # username
    passwd=config.conf['pymysql']['passwd'],        # password
    db=config.conf['pymysql']['db'])                # name of the database

# Create a Cursor object to execute queries.
cursor = db.cursor(pymysql.cursors.DictCursor)
cursor.execute("SELECT * FROM taiyuan_cityHour")
# cursor.execute("SELECT {}, pm25, {} FROM taiyuan_cityHour".format(feature1, feature2))
result_set = cursor.fetchall()
cursor.close()
result = pd.DataFrame(list(result_set))
