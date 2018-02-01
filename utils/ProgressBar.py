import sys
import os
import time
import numpy as np
from math import floor


class ProgressBar:
    '''
    Class to display the progress of the algorithm. It needs to be initialized at first and it has to be updated at each
    step. It displays the iterations per second, percentage of steps done until finish, and estimated time (ETA) to
    finish the progress on the format h:mm:s.ms.
    It can be added a custom message at the end of the display. It is usful to display losses, iteration number, etc.
    '''

    # TODO: If the message added is too large, the bar is not replaced correctly

    def __init__(self, nSteps, lenBar=50):
        '''
        Progress Bar initialization function
        :param nSteps: Number of updates/epochs that is going to be done
        :param lenBar: Number of characters for the progress bar
        '''
        self.__nSteps = nSteps + 1
        self.__lenBar = lenBar
        self.__step = 1
        self.__msg = None
        self.__start_time = time.time()
        self.__lenLastMsg = 0
        self.__vTimes = []
        self.__index = 0
        self.__loss = 0.0
        self.__funcMsg = self.__createProgressMsg if self.__lenBar > 0 else self.__createProgressMsgNoBar
        self.__lastMessg = ''
        self.__lastLens = []

    def update(self, loss=None, is_updated=True, show=True):
        '''
        Update the step and display the progress if needed.
        :param is_updated: Bool. True if the step has to increase. True by default
        :param show: Bool. True if the progress bar has to be displayed after the update. True by default.
        '''
        if self.__step <= self.__nSteps:
            if loss is not None:
                self.__loss = loss
            its = self.__updateTimes()
            progress_msg = self.__funcMsg(its)
            if show:

                self.__remove_last_msg()

                rows, columns = os.popen('stty size', 'r').read().split()
                msg_endl = progress_msg.split('\n')

                columns = int(columns) - 1

                self.__lastLens = []
                for msg in msg_endl:
                    init_msg = 0
                    len_msg = 0

                    if len(msg)==0:
                        msg_wrote = '\n'
                        self.__lastLens.append(len(msg_wrote))
                        sys.stdout.write(msg_wrote)
                    else:
                        while (init_msg < len(msg)):
                            end_msg = len_msg + min(len(msg) - init_msg, columns)
                            len_msg = end_msg - init_msg
                            if len_msg <0:
                                break

                            msg_wrote = msg[init_msg:end_msg] + '\n'
                            self.__lastLens.append(len(msg_wrote))
                            sys.stdout.write(msg_wrote)

                            init_msg = init_msg + len_msg

                sys.stdout.flush()

                if self.__step == self.__nSteps:
                    print ''

            if is_updated:
                self.__step = self.__step + 1
        else:
            print self.__step
            print self.__nSteps
            print 'WARNING: Progress Bar step is major than the limit established'

    def finish_progress(self):
        '''
        Function to finish the bar. It is needed to print something to put the pointer to the right point.
        '''
        print ''

    def set_msg(self, msg):
        '''
        Function to add a message. This message will be added at the end when displaying the bar.
        :param msg: String. Message
        '''
        self.__msg = msg

    def __remove_last_msg(self):

        if self.__lastLens != []:
            for indx, len_msg in enumerate(reversed(self.__lastLens)):

                sys.stdout.write('\x1b[1A')
                if len_msg>0:
                    sys.stdout.write(' ' * len_msg)
                    sys.stdout.write('\b' * len_msg)
            sys.stdout.flush()

    def __createProgressMsgNoBar(self, its):
        '''
        Private. Function to create the message Bar
        :param its: iteration per seconds
        :return: String. Return the progress message
        '''
        nRepetitions = int(float(self.__lenBar) * float(self.__step) / float(self.__nSteps))

        percentage = 100. * float(self.__step) / float(self.__nSteps)

        [hours, mints, sec] = self.__ETA(self.__nSteps, self.__step, its)

        loss = self.__loss

        progressMsg = '[' + '%.02f%%' % percentage + '], ' + '%.03f it/s, ' % its + \
                      ', ETA: %d:' % hours + '%02d:' % mints + "%02.01f" % sec
        if self.__msg != None:
            progressMsg = progressMsg + ', ' + self.__msg

        return progressMsg

    def __createProgressMsg(self, its):
        '''
        Private. Function to create the message Bar
        :param its: iteration per seconds
        :return: String. Return the progress message
        '''
        nRepetitions = int(float(self.__lenBar) * float(self.__step) / float(self.__nSteps))

        percentage = 100. * float(self.__step) / float(self.__nSteps)

        [hours, mints, sec] = self.__ETA(self.__nSteps, self.__step, its)

        loss = self.__loss

        progressMsg = '[' + ('=' * nRepetitions) + (' ' * (self.__lenBar - nRepetitions)) + \
                      '], ' + '%.03f it/s, ' % its + '%.02f%%' % percentage + \
                      ', ETA: %d:' % hours + '%02d:' % mints + "%02.01f" % sec
        if self.__msg != None:
            progressMsg = progressMsg + ', ' + self.__msg

        return progressMsg

    @staticmethod
    def __ETA(nSteps, step, its):
        '''
        Private. Compute the estimated time to finish the progress
        :param nSteps: Number of steps for the whole algorithm.
        :param step: Current step.
        :param its: Iterations per second.
        :return: List of floats. It contains the time on the format [hours, minutes, seconds]
        '''

        secs = (nSteps - step) / its
        mins = secs / 60.0
        hours = floor(mins / 60.0)
        mins = floor(mins - (hours * 60))
        secs = floor(secs - (mins * 60) - (hours * 60 * 60))

        return [hours, mins, secs]

    def __updateTimes(self):
        '''
        Private. Updates the time queue and compute the iterations per second.
        :return: Iterations per second.
        '''

        # Update time
        current_time = time.time()
        if len(self.__vTimes) >= 10:
            self.__vTimes[self.__index] = float(current_time - self.__start_time)
        else:
            self.__vTimes = np.append(self.__vTimes, float(current_time - self.__start_time))
        self.__start_time = current_time

        # Compute its
        its = 1.0 / np.mean(self.__vTimes)

        # Update step
        if self.__index >= 9:
            self.__index = 0
        else:
            self.__index = self.__index + 1

        return its

    def get_message(self,step=False):

        its = self.__updateTimes()
        progress_msg = self.__funcMsg(its) + '\n'

        if step:
            self.__step = self.__step + 1

        return progress_msg

'''
Testing class
'''

if __name__ == '__main__':


    print 'Testing small progress bar 5 seconds...'
    bar = ProgressBar(5, lenBar=0)
    bar.update(show=False)
    for i in range(5):
        time.sleep(1)
        bar.update()

    print 'Total testing estimated time...'
    global_bar = ProgressBar(5 + 30 + 5 + 5, lenBar=20)

    accum_str = '\n\nTesting large message during 5 seconds...\n'
    bar = ProgressBar(5)
    bar.update(show=False)
    for i in range(5):
        time.sleep(1)
        bar.set_msg('Testing msg after ' + str(
            i) + ' seconds with a veeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeery looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong message')
        last_str = bar.get_message(step=True)
        global_bar.set_msg(accum_str + last_str)
        global_bar.update()

    accum_str = accum_str + last_str

    accum_str =  accum_str + '\nTesting default progress bar 20 seconds...\n'
    n = 30
    bar = ProgressBar(n)
    bar.update(show=False)
    for i in range(n):
        time.sleep(1)
        last_str = bar.get_message(step=True)
        global_bar.set_msg(accum_str + last_str)
        global_bar.update()

    accum_str = accum_str + last_str

    accum_str = accum_str + '\nTesting double large progress bar 5 seconds...\n'
    bar = ProgressBar(5, lenBar=100)
    bar.update(show=False)
    for i in range(5):
        time.sleep(1)
        last_str = bar.get_message(step=True)
        global_bar.set_msg(accum_str + last_str)
        global_bar.update()

    accum_str = accum_str + last_str

    accum_str = accum_str + '\nTesting default progress bar with message during 5 seconds...\n'
    bar = ProgressBar(5)
    bar.update(show=False)
    for i in range(5):
        time.sleep(1)
        bar.set_msg('Testing msg after ' + str(i) + ' seconds')
        last_str = bar.get_message(step=True)
        global_bar.set_msg(accum_str + last_str)
        global_bar.update()

    accum_str = accum_str + last_str