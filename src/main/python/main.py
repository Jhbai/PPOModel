from src.main.python.common_func.logger import Log
from src.main.python.utils.Enviroment.Scenario1 import ENV
if __name__ == '__main__':
    logger = Log(__name__)
    env = ENV()
    logger.write('main started')