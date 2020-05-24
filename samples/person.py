#!/usr/bin/python3.6
class Person:
    color = 'blue'

    def __init__(self, color):
        self.name = name
        self.color = color

    def __init__(self, name, lname):
        self.name = name
        self.lname = lname

    def hello(self):
        print("hello", self.name, self.lname)

    def hi(self):
        print("hi", self.name, self.color)

def main():
    p = Person("V", "P")
    p.hello()
    p.hi()
    import keyword
    print(keyword.kwlist)

main()
