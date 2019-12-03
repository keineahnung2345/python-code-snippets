mport string
import itertools

# https://stackoverflow.com/questions/29351492/how-to-make-a-continuous-alphabetic-list-python-from-a-z-then-from-aa-ab-ac-e

def iter_all_strings():
    """
    a = itertools.count(1)
    next(a) # 1
    next(a) # 2
    """
    for size in itertools.count(1):
        for s in itertools.product(string.ascii_lowercase, repeat=size):
            yield "".join(s)

combs = []
max_length = 1000

for s in iter_all_strings():
    combs.append(s)
    if len(combs) == max_length:
        break

print(combs)
