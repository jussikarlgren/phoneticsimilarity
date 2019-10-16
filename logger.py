import inspect
import time

def logger(msg, level=False, endspace="\n"):
    if level:
        print(time.ctime(), " ", inspect.stack()[1][3], "(): ", msg, sep="", end=endspace, flush=True)
