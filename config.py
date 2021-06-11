class EnvConfig:
    ENV = "Pendulum-v0"

    def get_env_config(self, args):
        self.ENV = args['env'] if args['env'] else self.ENV


class AgentConfig:
    HIDDEN_SIZE = 256
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3
    GAMMA = 0.99    # Q'计算公式的γ
    TAU = 1e-2  # target网络软更新参数
    MAX_MEMORY_SIZE = 50000

    MAX_EPISODE = 100
    MAX_STEP = 1000
    BATCH_SIZE = 128

    EXPERIMENT_NO = 99  # 实验序号

    RES_PATH = './experiments/'  # parent folder storing the experiments' result

    def get_agent_config(self, args):
        self.HIDDEN_SIZE = args['hidden_size'] if args['hidden_size'] else self.HIDDEN_SIZE
        self.ACTOR_LR = args['actor_learning_rate'] if args['actor_learning_rate'] else self.ACTOR_LR
        self.CRITIC_LR = args['critic_learning_rate'] if args['critic_learning_rate'] else self.CRITIC_LR
        self.GAMMA = args['gamma'] if args['gamma'] else self.GAMMA
        self.TAU = args['tau'] if args['tau'] else self.TAU
        self.MAX_MEMORY_SIZE = args['max_memory_size'] if args['max_memory_size'] else self.MAX_MEMORY_SIZE

        self.MAX_EPISODE = args['max_episode'] if args['max_episode'] else self.MAX_EPISODE
        self.MAX_STEP = args['max_step'] if args['max_step'] else self.MAX_STEP
        self.BATCH_SIZE = args['batch_size'] if args['batch_size'] else self.BATCH_SIZE

        self.EXPERIMENT_NO = args['experiment_num'] if args['experiment_num'] else self.EXPERIMENT_NO