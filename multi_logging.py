#!/usr/bin/env python
import logging

out_dir = f'/home/finc/Dropbox/Projects/SelfReg/data/neuroimaging/1stlevel_survey/'


LOG_FILE_ONE = f"{out_dir}missing_fmri.log"
LOG_FILE_TWO = f"{out_dir}missing_events.log"
LOG_FILE_THREE = f"{out_dir}missing_confounds.log"
LOG_FILE_FOUR = f"{out_dir}high_motion.log"

def main():

    setup_logger('log_fmri', LOG_FILE_ONE)
    setup_logger('log_events', LOG_FILE_TWO)
    setup_logger('log_confounds', LOG_FILE_THREE)
    setup_logger('log_motion', LOG_FILE_FOUR)

    #logger('Logging out to log one...', 'info', 'one')
    #logger('Logging out to log two...', 'warning', 'two')

def setup_logger(logger_name, log_file, level=logging.INFO):

    log_setup = logging.getLogger(logger_name)
    #formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    formatter = logging.Formatter()
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)

def logger(msg, level, logfile):

    if logfile == 'log_fmri': log = logging.getLogger('log_fmri')
    if logfile == 'log_events'   : log = logging.getLogger('log_events')
    if logfile == 'log_confounds'   : log = logging.getLogger('log_confounds')
    if logfile == 'log_motion'   : log = logging.getLogger('log_motion')

    if level == 'info'    : log.info(msg)
    if level == 'warning' : log.warning(msg)
    if level == 'error'   : log.error(msg)

if __name__ == "__main__":

    main()
