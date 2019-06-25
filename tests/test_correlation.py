from nose.tools import *
import urllib
import sys
import pandas as pd

sys.path.append('../')
from bin.app import *
from lake.file import read as file_read
from mods import time_delayed_correlation_analysis
from mods import build_samples


class Test(object):
    def setup(self):
        starttime = '2017010101'
        endtime = '2017010123'
        self.data = build_samples.build_data_frame_for_correlation_analysis(starttime, endtime)

    def test_correlation(self):
        data = time_delayed_correlation_analysis.get_normalized_samples(self.data)
        assert_is_instance(data, pd.DataFrame)


if __name__ == '__main__':
    test = Test()
    test.setup()
    test.test_correlation()
