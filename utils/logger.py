import os
import sys


# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.log = open(log_file, "w")  # , 0)

    def write(self, message):
        self.log.write(message)
