import numpy as np
import os

fileName = 'q_table.txt'

try:
    os.remove(fileName)
except:
    pass

test = np.load('q_table.npy')
fo = open("q_table.txt", "a+")

string = " ".join(map(str, test))
fo.write(string + "\n")

fo.close()