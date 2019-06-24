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
        self.data = pd.read_csv("../tmp/taiyuan_cityHour.csv")
        self.data = json.load(open('./test_data/temp_high_mem.txt', 'rb'))

    def test_correlation(self):
        unitDataList = self.data['unitDataList']
        levelInfo = self.data['levelInfo']

        data_list = pd.DataFrame(unitDataList)
        value_levels = pd.DataFrame(levelInfo)
        samples = time_delayed_correlation_analysis(data_list, value_levels)



        assert_is_instance(image, PIL.Image.Image)
        image.save('../tmp/image.png')
    # image.show()
