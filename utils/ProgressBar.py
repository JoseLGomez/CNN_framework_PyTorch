import sys
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
        self.__nSteps       = nSteps+1
        self.__lenBar       = lenBar
        self.__step         = 1
        self.__msg          = None
        self.__start_time   = time.time()
        self.__lenLastMsg   = 0
        self.__vTimes       = []
        self.__index        = 0
        self.__loss         = 0.0
        self.__funcMsg      = self.__createProgressMsg if self.__lenBar>0 else self.__createProgressMsgNoBar


    def update(self, is_updated=True, show=True):
        '''
        Update the step and display the progress if needed.
        :param is_updated: Bool. True if the step has to increase. True by default
        :param show: Bool. True if the progress bar has to be displayed after the update. True by default.
        '''
        if self.__step  <= self.__nSteps:
            its             = self.__updateTimes()
            progress_msg    = self.__funcMsg(its)
            if show:
                sys.stdout.write('\b' * self.__lenLastMsg)
                sys.stdout.write(' ' * self.__lenLastMsg)
                sys.stdout.write('\b' * self.__lenLastMsg)
                sys.stdout.write(progress_msg)
                sys.stdout.flush()
                self.__lenLastMsg = len(progress_msg)
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

    def __createProgressMsgNoBar(self, its):
        '''
        Private. Function to create the message Bar
        :param its: iteration per seconds
        :return: String. Return the progress message
        '''
        nRepetitions = int(float(self.__lenBar)*float(self.__step)/float(self.__nSteps))

        percentage = 100.*float(self.__step)/float(self.__nSteps)

        [hours, mints, sec] = self.__ETA(self.__nSteps, self.__step, its)


        progressMsg = '[' + '%.02f%%' % percentage +  '], ' + '%.03f it/s, ' % its + \
                                ', loss: %.05f:' % loss  + \
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
                      ', loss: %.05f:' % loss + \
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
        if len(self.__vTimes)>=10:
            self.__vTimes[self.__index] = float(current_time - self.__start_time)
        else:
            self.__vTimes = np.append(self.__vTimes,float(current_time - self.__start_time))
        self.__start_time = current_time

        # Compute its
        its = 1.0/np.mean(self.__vTimes)

        # Update step
        if self.__index>=9:
            self.__index = 0
        else:
            self.__index = self.__index + 1

        return its


'''
Testing class
'''

if __name__ == '__main__':


    print 'Testing small progress bar 5 seconds...'
    bar = ProgressBar(5,lenBar=0)
    bar.update(show=False)
    for i in range(5):
        time.sleep(1)
        bar.update()

    print 'Testing large message during 5 seconds...'
    bar = ProgressBar(5)
    bar.update(show=False)
    for i in range(5):
        time.sleep(1)
        bar.set_msg('Testing msg after ' + str(i) + ' seconds with a veeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeery looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong message')
        bar.update()


    print 'Testing default progress bar 20 seconds...'
    n = 30
    bar = ProgressBar(n)
    bar.update(show=False)
    for i in range(n):
        time.sleep(1)
        bar.update()
    
    print 'Testing double large progress bar 5 seconds...'
    bar = ProgressBar(5,lenBar=100)
    bar.update(show=False)
    for i in range(5):        
        time.sleep(1)
        bar.update()    
    
    print 'Testing default progress bar 5 seconds and stop it...'
    bar = ProgressBar(5)
    bar.update(show=False)
    for i in range(2):        
        time.sleep(1)
        bar.update()
    bar.finish_progress()
        
    print 'Testing default progress bar with message during 5 seconds...'
    bar = ProgressBar(5)
    bar.update(show=False)
    for i in range(5):        
        time.sleep(1)
        bar.set_msg('Testing msg after '+ str(i) + ' seconds')
        bar.update()
