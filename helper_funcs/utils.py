from datetime import datetime
import sys
import torch

def find_last(pred, lst):
    return next(x for x in reversed(lst) if pred(x))

def t_stamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def temp_print(s):
    print (s + '\r'),
    sys.stdout.flush()


def zip_chunks(tensor, num_chunks, dim=0):
    return torch.stack(torch.chunk(tensor, num_chunks, dim), dim).transpose(dim,dim+1)