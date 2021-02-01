# Copyright 2020
# Author: Christian Leininger <info2016frei@gmail.com>
import os
import numpy as np


def time_format(sec):
    """
    Args:
        param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs, 2)


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def write_into_file(pathname, text):
    """ creates directory if needed an writes text in filename
    """
    mkdir(pathname, "")
    with open(pathname + ".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')
