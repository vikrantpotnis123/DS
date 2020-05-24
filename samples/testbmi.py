#!/usr/bin/python3.6

fobj = open("bmi.txt", "r")
for line in fobj:
   ht, wt =  line.strip().split(", ")
   print(ht,":",wt, ":",float(ht)/float(wt))
