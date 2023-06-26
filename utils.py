import logging
import socket
import csv
from contextlib import closing
from task import Task
import statistics


def make_logger(name):
    LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
    logger.addHandler(ch)

    return logger


def get_host_ip():
    """get the host ip elegantly
    https://www.chenyudong.com/archives/python-get-local-ip-graceful.html
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def find_free_port():
    """
    https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class Writer(object):
    def __init__(self, file_path) -> None:
        super().__init__()

        self.csvfile = open(file_path, 'w')
        fieldnames = ['record_time', 'job_id', 'model_name', 'iterations', 'batch_size', 'priority', 'throughput']
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames)
        self.writer.writeheader()
    

    def save(self, task: Task):
        row = {
            'record_time': task._timestamp,
            'job_id': task._job_id,
            'model_name': task._job_name,
            'iterations': task._iterations,
            'batch_size': task._batch_size,
            'priority': task._priority,
            'throughput': statistics.mean(task.throughputs[-5:-1]) / 10.,
        }
        self.writer.writerow(row)


    def close(self):
        self.csvfile.close()