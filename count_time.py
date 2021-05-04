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
def make_data(lst, interval):
    x = [] # 学習データ
    y = [] # 結果
    temps = lst
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

@count_time
def use_slice(lst, interval):
    train_data = []
    result = []

    for i in range(interval, len(lst)):
        result.append(lst[i])
        train_data.append(lst[i-interval:i])
    return (train_data, result)

@count_time
def use_inclusion(lst, interval):
    result = [lst[i] for i in range(interval, len(lst))]
    train_data = [lst[i-interval:i] for i in range(interval, len(lst))]
    return (train_data, result)

@count_time
def use_inclusion2(lst, interval):
    train_data = [lst[i-interval:i] for i in range(interval, len(lst))]
    result = [el[-1] for el in train_data]
    return train_data, result


sample_list1 = [f"t{i}" for i in range(1000)]
sample_list2 = [f"t{i}" for i in range(10000)]
sample_list3 = [f"t{i}" for i in range(100000)]
sample_list4 = [f"t{i}" for i in range(1000000)]

sample_lists = [sample_list1, sample_list2, sample_list3, sample_list4]

for lst in sample_lists:
    train_x, train_y = make_data(lst, 6)
    train_x, train_y = use_slice(lst, 6)
    train_x, train_y = use_inclusion(lst, 6)
    train_x, train_y = use_inclusion2(lst, 6)
