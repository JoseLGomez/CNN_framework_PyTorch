import os
import sys


# Save the printf to a log file
class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_on = False
        self.log = open(log_file, "a")  # , 0)

    def write(self, message):
        self.terminal.write(message)
        if self.log_on == True:
        	self.log.write(message)

    def log_stop(self):
    	self.log_on = False

    def log_start(self):
    	self.log_on = True

    def flush(self):
        pass
