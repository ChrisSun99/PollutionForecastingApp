from nose.tools import *
import urllib
import sys
import pandas as pd

sys.path.append('../')
from bin.app import *
from lake.file import read as file_read
from mods import time_delayed_correlation_analysis


class Test(object):
    def setup(self):
        csvfile = open('../tmp/taiyuan_cityHour.csv', 'r')
        jsonfile = open('../tmp/data.json', 'w')
        fieldnames = (
            '_class', '_id', 'aqi', 'ci', 'city', 'co', 'coIci', 'grade', 'no2', 'no2Ici', 'o3', 'o3H8', 'o3Ici',
            'pm10',
            'pm10Ici', 'pm25', 'pm25Ici', 'pp', 'ptime', 'so2', 'so2Ici', 'sd', 'temp', 'wd', 'weather', 'ws', 'itime',
            "regionId")
        reader = csv.DictReader(csvfile, fieldnames)
        for row in reader:
            json.dump(row, jsonfile)
            jsonfile.write('\n')
        self.data = []
        with open('../tmp/data.json', 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def test_correlation(self):
        data = time_delayed_correlation_analysis.get_normalized_samples(self.data)
        assert_is_instance(data, dict)
        data.save('../tmp/total_ccf_results.pkl')
