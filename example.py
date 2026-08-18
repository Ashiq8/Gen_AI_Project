def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper

@decorator
def say_whee():
    """This prints Whee!"""
    print("Whee!")

print(say_whee.__name__)   # wrapper
help(say_whee)             # shows wrapper info, not say_whee
