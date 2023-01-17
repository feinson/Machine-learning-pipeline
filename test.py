class Person:

    def __init__(self, age):
        self.age = age

    def __str__(self):
        return f"person's age is {self.age}"


gill = Person(5)
print(gill)