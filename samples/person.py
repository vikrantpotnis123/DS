#!/usr/bin/python3.6
class Person:
    color = 'blue'

    def __init__(self, name, color):
        self.name = name
        self.color = color

    def __init__(self, name, lname):
        print('init')
        self.name = name
        self.lname = lname

    def __enter__(self):
        # print("hello", self.name, self.lname)
        print("enter")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print("exit")

    def hello(self):
        print("hello", self.name, self.lname)

    def hi(self):
        print("hi", self.name, self.color)

def main():
    '''
    p = Person("V", "P")
    p.hello()
    p.hi()
    '''

    import keyword
    print(keyword.kwlist)

    with Person('p', 'p') as p1:
        p1.hello()


main()
