from datetime import datetime
import sys

def find_last(pred, lst):
    return next(x for x in reversed(lst) if pred(x))

def t_stamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def temp_print(s):
    print (s + '\r'),
    sys.stdout.flush()