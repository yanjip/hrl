import logging
from typing import List

from termcolor import colored


class ProjectLogger:
    
    def __init__(self,
                 log_file: str = None,
                 level: int = logging.DEBUG,
                 printing: bool = True, attrs: List[str] = None,
                 name: str = 'project_logger',
                 ):
        """ Basic logger that can write to a file on disk or to sterr.

        :param log_file: name of the file to log to
        :param level: logging verbosity level
        :param printing: flag for whether to log to sterr
        """
        root_logger = logging.getLogger(name)
        root_logger.setLevel(level)
        self.printing = printing
        self.attrs = attrs
        
        # Set up writing to a file
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_formatter = logging.Formatter(
                '%(levelname)s: %(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        # Set up printing to stderr
        def check_if_sterr(hdlr: logging.Handler):
            return isinstance(hdlr, logging.StreamHandler)\
                   and not isinstance(hdlr, logging.FileHandler)
        
        if printing and not list(filter(check_if_sterr, root_logger.handlers)):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            root_logger.addHandler(console_handler)
        
        self.log = root_logger
    
    def debug(self, msg, color='grey', attrs: List[str] = None):
        self.log.debug(colored(msg, color, attrs=attrs or self.attrs))
    
    def info(self, msg, color='green', attrs: List[str] = None):
        self.log.info(colored(msg, color, attrs=attrs or self.attrs))
    
    def warning(self, msg, color='blue', attrs: List[str] = None):
        self.log.warning(colored(msg, color, attrs=attrs or self.attrs))
    
    def error(self, msg, color='magenta', attrs: List[str] = None):
        self.log.error(colored(msg, color, attrs=attrs or self.attrs))
    
    def critical(self, msg, color='red', attrs: List[str] = None):
        self.log.critical(colored(msg, color, attrs=attrs or self.attrs))
