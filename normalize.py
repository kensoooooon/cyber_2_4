# 正規化
from sklearn import preprocessing

sample = [[ 3, -2,  8],
          [ 2.1,  9.2,  4.4],
          [ 0.2,  3.5, 1.1],
          [2.5, -3.3, 0.56],
          [33.2, -25.6, 13.4],
          [23, 32, -1.8],
         ]
sample_normalized = preprocessing.normalize(sample, norm='l2')
print(sample_normalized)