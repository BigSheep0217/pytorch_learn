import cv2
import os
import argparse
from tqdm import tqdm
# import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import Lock, Pool, Queue
import multiprocessing as mp

import numpy as np


from utils import *


def my_print1(l):
    l.acquire()
    try:
        print(os.getpid())
    finally:
        l.release()
    time.sleep(1)
    # return os.getpid()


if __name__ == '__main__':
    
    lock = mp.Manager().Lock() # 进程池的进程不是由当前同一个父进程创建，所以需要借用Manager进行跨进程的管理
    # task_Q = mp.Manager().Queue()

    MutiProcess_pool = mp.Pool(processes=os.cpu_count())    
    results = []
    for _ in range(10):
        results.append(MutiProcess_pool.apply_async(my_print1, args=(lock, ))) # 多个参数，但是不能迭代，只能一个一个加
        
    print(f"results : {len(results)}")
    for result in tqdm(results):
        result.get() # 阻塞
        
    MutiProcess_pool.close()
    MutiProcess_pool.join()
    
    

    
    


