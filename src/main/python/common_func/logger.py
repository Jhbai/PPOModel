import logging
from datetime import datetime
class Log:
    def __init__(self, name):
        # Naming Extraction
        name = name.split('.')[-1].replace('_', '')
        # Set Up logging object
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # Logging info save to txt
        File_Handler = logging.FileHandler(f'./src/main/python/common_func/LOG/{name}.txt', mode = 'w+')
        self.logger.addHandler(File_Handler)

    def write(self, msg):
        self.logger.info(f'[{datetime.now().replace(microsecond = 0)}]{msg}')