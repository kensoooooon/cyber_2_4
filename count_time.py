#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from functools import wraps
import time

def count_time(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.perf_counter()
        result = func(*args, **kargs)
        elapsed_time = time.perf_counter() - start
        print(f"{func.__name__}は{elapsed_time}秒かかりました")
        return result
    return wrapper

@count_time
def use_loop(lst, interval):
    train_data = []
    result = []
    
    for i in range(len(lst)):
        if i < interval:
            continue
        result.append(lst[i])
        train_data_sub = []
        for p in range(interval):
            d = i + p - interval
            train_data_sub.append(lst[d])
        train_data.append(train_data_sub)
    return (train_data, result)

@count_time
def use_slice(lst, interval):
    train_data = []
    result = []

    for i in range(interval, len(lst)):
        result.append(lst[i])
        train_data.append(lst[i-6:i])
    return (train_data, result)


sample_list = [f"t{i}" for i in range(100000)]

train_x, train_y = use_loop(sample_list, 6)
train_x, train_y = use_slice(sample_list, 6)

