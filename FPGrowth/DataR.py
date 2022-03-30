import matplotlib.pyplot as plt

import numpy as np
from collections import OrderedDict
import csv

def data_preprocess(doc_name, threshold):
    data = np.genfromtxt(doc_name, delimiter=file_delimiter, dtype=str)
    freq = {}
    # 1st database scan
    for (x, y), value in np.ndenumerate(data):
        # Check if the item is a null value or not.
        # If not, append it to the  freq dictionary.

        if value != '' or '?' or 0:
            if value not in freq:
                freq[value] = 1
            else:
                freq[value] += 1
    # Removing items whcih are below the given support value
    freq = {k: v for k, v in freq.items() if v >= threshold}
    return data, freq
support =9
file_name = 'Small1.csv'

file_delimiter = ','   #this has to be changed according to the data set decription(for most cases its ',')
dataset, freq_items = data_preprocess(file_name, support)
print(freq_items)
plt.bar(range(len(freq_items)), list(freq_items.values()))
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()