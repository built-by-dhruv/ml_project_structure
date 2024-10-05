import logging
import sys

def error_message_detail(error, error_detail:sys):
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    fname = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error Type: {exc_type}\nError Message: {exc_obj}\nError Line: {exc_tb.tb_lineno}\nError File: {fname}"
    return error_message


class CustomException(Exception):
    def __init__(self, error_msg , error_msg_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg,error_detail= error_msg_detail)
    
    def __str__(self):
        return self.error_msg
    
if __name__ == '__main__':
    try:
        # raise CustomException('This is a custom exception', sys)
        a=1/0
    except Exception as e:
        logging.info('Divide by zero error')
        raise CustomException(e, sys)