#!/usr/bin/python3.6


fobj = open("ct.txt", "r")
l = []
for line in fobj:
   *city, day, time = line.split()
   h, m = time.split(":")
   l.append((" ".join(city), day, ( int(h), int(m) ) ))
l.sort()
print(l)
