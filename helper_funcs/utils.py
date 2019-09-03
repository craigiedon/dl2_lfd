from datetime import datetime
import json
import torch
from collections import defaultdict

def find_last(pred, lst):
    return next(x for x in reversed(lst) if pred(x))

def t_stamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def zip_chunks(tensor, num_chunks, dim=0):
    return torch.stack(torch.chunk(tensor, num_chunks, dim), dim).transpose(dim,dim+1)

def temp_print(s):
    return


def load_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data

def load_json_lines(file_path):
    data = defaultdict(list)
    with open(file_path, 'r') as fp:
        for line in fp:
            j_dict = json.loads(line)
            for k, v in j_dict.items():
                data[k].append(v)
    return data
