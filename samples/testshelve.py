#!/usr/bin/python3.6

import shelve

phbooks = shelve.open("myphbook")
phbooks["ph1"] =  {"nm1" : "a1", "nm2" : "a2" }
phbooks["ph2"] =  {"nm3" : "a3", "nm4" : "a4", "nm3" :  "a33" }
phbooks.close()

phbooks2 = shelve.open("myphbook")
print(phbooks2["ph2"]["nm3"])

