# coding='utf-8'

import datetime


def stamp():
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d %H:%M:%S")

    return StyleTime


if __name__ == '__main__':
    print stamp()