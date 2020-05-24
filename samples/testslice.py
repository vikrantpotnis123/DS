#!/usr/bin/python3.6

l = [ "i1", "i2", "i3", "i4", "i5", "i6" ]
s = l[2:4:2]
print(s)
s = l[5:3:-2]
print(s)

n1 = [ 1, 2, 3 ]
n2 = n1
print(id(n1), id(n2))
print(n1, ":", n2)

n2[1] = 88
print(id(n1), id(n2))
print(n1, ":", n2)

n2 = [ 4, 5, 6 ]
print(id(n1), id(n2))
print(n1, ":", n2)

n3  = n2[:]
print(n2, ":", n3)

n3 = [ 99, 99, 99 ]
print(n2, ":", n3)

n3[2] = 55
print(n2, ":", n3)

no = [ '4', '5', '6', [ '1', '2', '3' ] ]
yo = no[:]
no[3][0] = '22'

print(no, ":", yo)

from copy import deepcopy as dpc
no = [ 1, 2, 3, [2, 3, 4] ]
yes = dpc(no)
yes[3][2] = 33

print(no, ":", yes)
