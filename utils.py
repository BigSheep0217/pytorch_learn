import time
import datetime


def get_million_time_str(prefix='', suffix=''):
    time.sleep(0.001)
    date_time = datetime.datetime.now().strftime('%y%m%d%H%M%S')
    ms = str(int(time.time()*1000)%1000)
    return prefix + date_time + ms + suffix