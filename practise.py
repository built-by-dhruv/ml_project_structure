class A:
    a = 10
    def print(self):
        self.a = 20
        print("Hello")
        
class B:
    def __init__(self):
        self.check = A()
        self.check.a = 20
    def display(self):
        print(self.check.a)

a = B() 
a.display()




