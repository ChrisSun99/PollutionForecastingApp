
import pymysql
import pandas as pd
import numpy as np
from analysis import time_delayed_correlation_analysis
from mods import data_filtering
import matplotlib.pylab as plt
import seaborn as sns
import sys
import csv

from bin import app

sys.path.append('../')

from analysis.acf_pacf_test import acf_pacf_test
pymysql.install_as_MySQLdb()

db = pymysql.connect(
    host="localhost",  # your host
    user="root",       # username
    passwd="Smt19990513!",     # password
    db="sample_db")   # name of the database

# Create a Cursor object to execute queries.
cur = db.cursor()

# Select data from table using SQL query.
#cur.execute("SELECT * FROM beijing_cityhour")
fields = pd.read_csv("../config/nameList.csv")
data = [row for row in csv.reader(fields)]
cursor = db.cursor(pymysql.cursors.DictCursor)
feature1 = data[0][0]
feature2 = data[1][0]
cursor.execute("SELECT {}, pm25, {} FROM beijing_cityHour".format(feature1, feature2))
result_set = cursor.fetchall()
# for row in result_set:
#     print("%s, %s, %s" % (row["ptime"], row["pm25"], row["aqi"]))
cur.close()

result = pd.DataFrame(result_set)

NON_DER = ['aqi', ]
df_new = time_delayed_correlation_analysis.df_derived_by_shift(result, 6, NON_DER)

"""
可视化
"""
colormap = plt.cm.RdBu
plt.figure(figsize=(15, 10))
plt.title(u'6 days', y=1.05, size=16)

mask = np.zeros_like(df_new.corr())
mask[np.triu_indices_from(mask)] = True

svm = sns.heatmap(df_new.corr(), mask=mask, linewidths=0.1, vmax=1.0,
                  square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()