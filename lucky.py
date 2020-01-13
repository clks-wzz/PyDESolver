import os

def subproc():
    print('lucky miaomiao!')

def procfun(fun):
    fun()

def process():
    procfun(subproc)

if __name__ == '__main__':
    process()