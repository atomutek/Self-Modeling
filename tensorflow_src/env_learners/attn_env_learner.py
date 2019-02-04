
from tensorflow_src.env_learners import EnvLearner
class AttnEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
