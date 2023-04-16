import logging
import sys
import logger

class ProjectException(Exception):
    def __init__(self, message,detail : sys):
        super().__init__(message)

        exc_type, exc_obj, exc_tb = detail.exc_info()
        self.message = "Error " + str(exc_tb.tb_frame.f_code.co_filename) + " line no. " + str(exc_tb.tb_lineno) + " " + str(message)

    def __str__(self):
        return self.message

if __name__ == '__main__':

    try:
        1/0
    except ZeroDivisionError as e:
        test = ProjectException(e,sys)
        logging.info(str(test))