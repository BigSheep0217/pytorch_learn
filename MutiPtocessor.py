import cv2
import os
import argparse
from tqdm import tqdm
# import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Pool
import numpy as np


from utils import *


def my_print1(a, b):
    print(a+b)
    time.sleep(1)

def my_print2(a):
    print(a)
    time.sleep(1)
    
a = [1,2,3,4,5,6,7,8]
b = [1,2,3,4,5,6,7,8]

if __name__ == '__main__':
    
    cpu_count = os.cpu_count()

    MutiProcess_pool = Pool(processes=cpu_count)
    
    for i, k in zip(a, b):
        MutiProcess_pool.apply_async(my_print1, (i, k)) # 多个参数，但是不能迭代，只能一个一个加
    MutiProcess_pool.close()
    MutiProcess_pool.join()
    
    # MutiProcess_pool.map_async(my_print2, a) # 单个参数，可迭代
    


