from src.main.python.common_func.logger import Log
def ENV():
    logger = Log(__name__)
    logger.write('Create Enviroment')
    try:
        logger.write('Package Exist')
        import gym
        env = gym.make("LunarLander")
    except:
        logger.write('Package Does not Exist, Package Installing')
        import os
        os.system('pip install gym')
        os.system('pip install box2d-py')
        os.system('pip install box2d')
        os.system('pip install Box2D-2.3.2-cp39-cp39-win_amd64.whl')
        os.system('pip install pygame')
        import gym
        env = gym.make("LunarLander")
    return env