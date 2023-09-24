import numpy as np
import pandas as pd
import tensorflow as tf

p = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]]

arr = tf.constant(p)

print(arr)
print(arr.shape)



