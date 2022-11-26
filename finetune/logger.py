#!/usr/bin/python  
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import io


def create_detail_day():
    '''
    :return:
    '''
    daytime = datetime.now().strftime('day'+'%Y_%m_%d')
    detail_time = daytime
    return detail_time


class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    start_time = datetime.now()

    sys.stdout = Logger('result.log', path='./')
    print('start time {}'.format(start_time))
    print(create_detail_day().center(60, '*'))
    print('explanation'.center(80, '*'))
    info1 = 'sort the form large to small'
    info2 = 'sort the form large to small'
    print(info1)
    print(info2)
    print('END: explanation'.center(80, '*'))

    end_time = datetime.now()
