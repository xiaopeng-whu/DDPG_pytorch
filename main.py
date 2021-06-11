from agent import Agent


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Add arguments for DDPG_pytorch configs')
    parser.add_argument('-x', '--experiment_num', metavar='N', type=int, help='experiment number', required=True)
    parser.add_argument('-hs', '--hidden_size', metavar='hs', type=int, help='hidden size')
    parser.add_argument('-alr', '--actor_learning_rate', metavar='alr', type=float, help='actor learning rate')
    parser.add_argument('-clr', '--critic_learning_rate', metavar='clr', type=float, help='critic learning rate')
    parser.add_argument('--gamma', metavar='g', type=float, help='gamma')
    parser.add_argument('--tau', metavar='g', type=float, help='tau')
    parser.add_argument('-mms', '--max_memory_size', metavar='mms', type=int, help='max memory size')

    parser.add_argument('-ms', '--max_step', metavar='ms', type=int, help='max step')
    parser.add_argument('-mepi', '--max_episode', metavar='mepi', type=int, help='max number of episodes')
    parser.add_argument('-b', '--batch_size', metavar='bs', type=int, help='batch size')

    parser.add_argument('--env', metavar='env', type=str, help='environments')
    return parser.parse_args()


if __name__ == '__main__':
    args = vars(get_args())
    agent = Agent(args)
    agent.train()
    # agent.env_close()
    agent.save_results()