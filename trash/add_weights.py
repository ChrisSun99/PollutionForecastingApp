# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""
import torch
from torch import nn
import numpy as np


if __name__ == '__main__':
	a, b = np.array([[1, 1, 1, 1], [3, 4, 3, 3]]), np.array([[2, 2, 2, 2], [5, 5, 5, 5]])
	a, b = torch.from_numpy(a.astype(np.float32)), torch.from_numpy(b.astype(np.float32))
	
	weights_len = a.shape[1]
	weights = [1.0]
	for i in range(1, weights_len):
		weights.append(pow(i + 1, 2))
	
	weights = np.array(weights).astype(np.float32).reshape(1, -1)
	weights = torch.from_numpy(weights)
	
	
	l1 = nn.L1Loss()
	loss = l1(weights * a, weights * b)


