import sys
from src.spotify_song_explorer.logger import logging

def error_message_detail(error_message, error_details: sys):
    exc_type, exc_value, exc_tb = error_details.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{error_message}]"
    else:
        # Fallback when no traceback available
        return f"Error message: {error_message}"

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        return self.error_message
