#!/usr/bin/python3.6
import array as arr
def testarray1(a):
    print(a)
    arr.array = [10, 20, 30]
    print(arr.array)
    arr.array += a
    arr.array1 = [[2,3], [4,5]]
    print(arr.array1[0][1])
    return arr.array


def main():
    print(testarray1([1]))
    print(testarray1([2]))

main()
