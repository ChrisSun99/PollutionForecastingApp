
import pymysql
import pandas as pd
import numpy as np
from analysis import time_delayed_correlation_analysis
from mods import data_filtering
import matplotlib.pylab as plt
import seaborn as sns
import sys

sys.path.append('../')

from analysis.acf_pacf_test import acf_pacf_test
from mods.config_loader import config

if __name__ == '__main__':
    # Connect DB
    pymysql.install_as_MySQLdb()
    db = pymysql.connect(
        host = config.conf['local_db']['host'],         # your host
        user = config.conf['local_db']['user'],         # username
        passwd = config.conf['local_db']['passwd'],     # password
        db = config.conf['local_db']['db'])             # name of the database
    
    cur = db.cursor()                                   # Create a Cursor object to execute queries.
    
    # Select data from table using SQL query.
    #cur.execute("SELECT * FROM beijing_cityhour")
    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT ptime, pm25, aqi FROM beijing_cityHour")
    result_set = cursor.fetchall()
    # for row in result_set:
    #     print("%s, %s, %s" % (row["ptime"], row["pm25"], row["aqi"]))
    cur.close()
    
    # Extract data from query results
    result = pd.DataFrame(result_set)
    
    NON_DER = ['aqi', ]
    df_new = time_delayed_correlation_analysis.df_derived_by_shift(result, 6, NON_DER)
    
    # Data visualizing
    colormap = plt.cm.RdBu
    plt.figure(figsize=(15, 10))
    plt.title(u'6 days', y=1.05, size=16)
    
    mask = np.zeros_like(df_new.corr())
    mask[np.triu_indices_from(mask)] = True
    
    svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=0.1, vmax=1.0,
                      square=True, cmap=colormap, linecolor='white', annot=True)
    
    plt.show()