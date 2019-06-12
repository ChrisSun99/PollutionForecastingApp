
import pymysql
import pandas as pd

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
cur.execute("SELECT * FROM beijing_cityhour")

#Record the data
data_bj = list(cur.fetchall())
cur.close()

df_bj = pd.DataFrame(data_bj)

# # print the first and second columns
# # for row in data:
# #     print(row[0], " ", row[1])
print(df_bj)
# acf_pacf_test(beijing_cityhour, data);