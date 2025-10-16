import scipy.io as sio
import numpy as np

# 加载 mat 文件
data_train = sio.loadmat('data_train.mat')
data_test = sio.loadmat('data_test.mat')
label_train = sio.loadmat('label_train.mat')

print('=== data_train.mat 内容 ===')
print('Keys:', list(data_train.keys()))
for key in data_train.keys():
    if not key.startswith('__'):
        print(f'{key}: shape = {data_train[key].shape}, dtype = {data_train[key].dtype}')
        print(f'Preview:\n{data_train[key][:5]}')

print('\n=== label_train.mat 内容 ===')
print('Keys:', list(label_train.keys()))
for key in label_train.keys():
    if not key.startswith('__'):
        print(f'{key}: shape = {label_train[key].shape}, dtype = {label_train[key].dtype}')
        print(f'Preview:\n{label_train[key][:10]}')

print('\n=== data_test.mat 内容 ===')
print('Keys:', list(data_test.keys()))
for key in data_test.keys():
    if not key.startswith('__'):
        print(f'{key}: shape = {data_test[key].shape}, dtype = {data_test[key].dtype}')
        print(f'Preview:\n{data_test[key][:5]}')