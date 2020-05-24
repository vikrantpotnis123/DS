#!/usr/bin/python3.6

import pickle

f = open("/tmp/data.pkl", "bw")
m = { 'm1' : 1, 'm2' : 2}
pickle.dump(m, f)
f.close()

f = open("/tmp/data.pkl", "ba")
m = { 'm3' : 3, 'm4' : 4}
pickle.dump(m, f)
f.close()

f = open("/tmp/data.pkl", "br")
n = pickle.load(f)
o = pickle.load(f)
print(n)
print(o)
f.close()
