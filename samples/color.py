#!/usr/bin/python3.6
import trees
class Person:
    color = 'blu'
    def __init__(self, name, lname):
        self.name = name
        self.lname = lname

    def hello(self):
        print("hello", self.name, self.lname)

    def hi(self):
        print("hi", self.name, self.color)

class P:
    def __init__(self, x):
        self.__x = x

    def __get_x(self):
        return self.__x

    def __set_x(self, x):
        self.__x = x

class PersonInfo:
    @property
    def full(self):
        return self.first + " " + self.last

    def __init__(self, first, last, email):
        self.first = first
        self.last = last
        self.email = email

    def get_info(self):
        print(self.full, ":", self.email)


def main():
    p = Person('v', 'p')
    p.hello()
    p.hi()
    p1 = P(1)
    pi = PersonInfo("v", "p", "vp@gmail.com")
    pi.get_info()

    pi.first = "p"
    pi.last = "p"

    pi.get_info()

if __name__ == '__main__':
    exit(main())
