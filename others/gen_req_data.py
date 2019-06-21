# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

生成请求数据
"""
import json


def gen_req_data(start_time, end_time = ''):
	"""
	生成请求数据
	:param start_time: str, 查询起始时间, 如'2017010101'
	:param end_time: str, 查询结束时间, 如'2017010123'
	:return: req_dict
	"""
	req_dict = {
		'start_time': start_time,
		'end_time': end_time
	}
	with open('../tmp/request.pkl', 'w') as f:
		json.dumps(req_dict)
	return req_dict


if __name__ == '__main__':
	start_time = '2017010101'
	end_time = '2018010101'
	req_data = gen_req_data(start_time, end_time)
	
	
	

