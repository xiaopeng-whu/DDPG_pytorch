import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from networks import *
from utils import *
from config import AgentConfig, EnvConfig


class Agent(AgentConfig, EnvConfig):
    def __init__(self, args):
        # self.env = env
        self.get_env_config(args)
        self.get_agent_config(args)
        self.env = NormalizedEnv(gym.make(self.ENV))
        # Params
        self.num_states = self.env.observation_space.shape[0]  # number of states
        self.num_actions = self.env.action_space.shape[0]  # number of actions
        # self.gamma = gamma  # Q'计算公式的γ
        # self.tau = tau  # target网络软更新参数

        # Networks
        hidden_size = self.HIDDEN_SIZE
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)  # neural network of input num_states, hidden layer size hidden_size and output num_actions
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)  # same as above
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)  # neural network of input num_states + num_actions, hidden layer size hidden_size and output num_actions
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.replay_buffer = ReplayBuffer(self.MAX_MEMORY_SIZE)  # inizialize replay_buffer of size max_memory_size
        self.critic_loss = nn.MSELoss()  # function for calculate loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ACTOR_LR)  # optimizer for gradient descent
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.CRITIC_LR)

        # Noise
        self.noise = OUNoise(self.env.action_space)  # noise to add on actions

    # from a state get a specific action
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))  # unsqueeze = view()
        # action = self.actor(state) 也可以通过内置call函数直接调用forward函数
        action = self.actor.forward(state)  # call forward fuction that use relu activation on neural network
        action = action.detach().numpy()[0, 0]  # take first action in first row first column
        return action

    def update(self):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(
            self.BATCH_SIZE)  # take states, actions, rewards and next_states from a sample of memory ReplayBuffer
        states = torch.FloatTensor(states)  # transform a list in a FloatTensor --> a tensor of float
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss --> we want to minimize the loss function
        Qvals = self.critic.forward(states, actions)  # the q value output of critic network
        next_actions = self.actor_target.forward(next_states)  # the actions from output of actor network
        next_Q = self.critic_target.forward(next_states,
                                            next_actions.detach())  # combine actor(next_actions) network and critic network(obs next from replay_buffer) --> q-val next
        Qprime = rewards + self.GAMMA * next_Q  # calculate Qprime

        critic_loss = self.critic_loss(Qvals, Qprime)  # the difference between qvalue - q'

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()   # maximize the q value calculating from critic where action is taken from actor

        # update networks
        self.actor_optimizer.zero_grad()  # gradient descent
        policy_loss.backward()  # backpropagation
        self.actor_optimizer.step()  # step

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.TAU + target_param.data * (1.0 - self.TAU))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.TAU + target_param.data * (1.0 - self.TAU))

    def train(self):
        self.episode_durations = []  # 记录每个episode的持续step数
        self.rewards = []

        for episode in range(self.MAX_EPISODE):
            self.noise.reset()
            state = self.env.reset()
            episode_reward = 0

            for step in range(self.MAX_STEP):
                # env.render()
                action = self.get_action(state)
                action = self.noise.get_action(action, step)
                new_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, new_state, done)

                if len(self.replay_buffer) > self.BATCH_SIZE:
                    self.update()

                state = new_state
                episode_reward += reward

                if done:
                    self.episode_durations.append(step + 1)
                    print("episode " + str(episode) + " finished after " + str(step + 1) + " timesteps, reward " + str(episode_reward))
                    break

            self.rewards.append(episode_reward)

        return self.rewards

    def save_results(self):
        # plot and save figure
        plt.figure(0)
        # policy_net_scores = torch.tensor(self.policy_net_scores, dtype=torch.float)
        plt.title("DDPG Experiment %d" % self.EXPERIMENT_NO)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(self.rewards)  # 当前episode对应的reward
        # Take 10 episode policy net score averages and plot them too

        plt.savefig(self.RES_PATH + "%d-result.png" % self.EXPERIMENT_NO)
        # plt.show()

        self.write_results(self.RES_PATH)

    def write_results(self, PATH):
        attr_dict = {
            "HIDDEN_SIZE": self.HIDDEN_SIZE,
            "ACTOR_LR": self.ACTOR_LR,
            "CRITIC_LR": self.CRITIC_LR,
            "GAMMA": self.GAMMA,
            "TAU": self.TAU,
            "MAX_MEMORY_SIZE": self.MAX_MEMORY_SIZE,

            "MAX_EPISODE": self.MAX_EPISODE,
            "MAX_STEP": self.MAX_STEP,
            "BATCH_SIZE": self.BATCH_SIZE,

            "EXPERIMENT_NO": self.EXPERIMENT_NO,

            "RES_PATH": self.RES_PATH
        }
        with open(PATH + "%d-log.txt" % self.EXPERIMENT_NO, 'w') as f:
            for k, v in attr_dict.items():
                f.write("{} = {}\n".format(k, v))
            f.write("------------------\n")
            for i in range(len(self.rewards)):
                f.write("Ep %d finished after %d steps -- reward: %.2f\n"
                        % (
                        i + 1, self.episode_durations[i], self.rewards[i]))
