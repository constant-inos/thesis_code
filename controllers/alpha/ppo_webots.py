import gym
import tensorflow as tf
from collections import deque
import os
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import shelve

from PPO import Agent
from WebotsEnv import Mitsos

from networks import *
from experience_memory import *

class Memory(Memory):
    def __init__(self,n_actions):
        super(Memory,self).__init__(n_actions=n_actions)

    def read_memory(self):
        samples = self.memory
        batch_size = self.memCounter
        
        num_lists = len(self.memory[0])
        lists = [[] for _ in range(num_lists+2)]
        
        for sample in samples:
            image = sample[0][0]
            image_ = sample[3][0]
            sensors = sample[0][1]
            sensors_ = sample[3][1]
            sample = [image,sensors] + list(sample[1:3]) + [image_,sensors_] + list(sample[4:])
            for i in range(len(sample)):
                lists[i].append(sample[i])

        
        lists = [np.vstack(l) if isinstance(l[0],np.ndarray) else np.vstack(l).reshape(-1) for l in lists ]

        lists = [[lists[0],lists[1]]] + lists[2:4] + [[lists[4],lists[5]]] + lists[6:]

        return tuple(lists)

class DoubleInputAgent(Agent):
    def __init__(self,n_actions,lr,gamma=0.99):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.LAMBDA_GAE = 0.95
        self.PPO_EPOCHS = 10
        self.MINIBATCH_SIZE = 64
        self.PPO_EPSILON = 0.15

        self.ppo_network = MitsosPPONet(n_actions)
        self.ppo_network.compile(optimizer=Adam(lr=lr))

        self.memory = Memory(n_actions=n_actions)

    def choose_action(self,state):
        state = [ tf.convert_to_tensor([state[0]]) , tf.convert_to_tensor([state[1]]) ]
        probs, value = self.ppo_network(state)
        action_dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        action = action_dist.sample()
        try:
            log_prob = action_dist.log_prob(action)
        except:
            print('Nan Error!')            
            exit()

        return int(action.numpy()[0]), log_prob.numpy()[0], value.numpy()[0][0]


env = Mitsos()
agent = DoubleInputAgent(n_actions=env.action_size,lr=0.0005)


RESTORE_DAMAGE = 30
n_games = 2000
discrete_actions = [[-1,1],[1,1],[1,-1]]
training = True
k = -1
filename = 'saveforreload.out'


scores = deque(maxlen=100)
i = 0


if os.path.exists(filename):
    my_shelf = shelve.open(filename)
    for key in my_shelf:
        globals()[key]=my_shelf[key]
    my_shelf.close()
    del my_shelf
    print('VARIABLES LOADED')
    os.remove(filename)



def test_agent(env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action,_,_ = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        state = state_
        total_reward += reward
    return total_reward

TEST_EPOCHS = 5
PPO_STEPS = 256
TARGET_SCORE = 200

train_epochs = 0
early_stop = False

while not early_stop:
    observation = env.reset()
    for _ in range(PPO_STEPS):
        action, log_probs, value = agent.choose_action(observation) #observation = [frames_stack, sensors_stack]
        observation_, reward, done, info = env.step(action)
        
        state = [np.expand_dims(observation[0],axis=0),np.expand_dims(observation[0],axis=0)]
        state_ = [np.expand_dims(observation_[0],axis=0),np.expand_dims(observation_[0],axis=0)]
        agent.store_experience(state,action,reward,state_,done,log_probs,value)
        if done:
            observation = env.reset()
            continue
        observation = observation_

    obs = [ tf.convert_to_tensor([observation_[0]]) , tf.convert_to_tensor([observation_[1]]) ]
    _,next_value = agent.ppo_network(obs)
    next_value = next_value.numpy()[0][0]
    states,actions,rewards,states_,dones,log_probs,values = agent.read_memory()#?
    returns = agent.compute_gae(next_value,values,rewards,dones)
    advantages = returns - values
    agent.ppo_update(states,actions,log_probs,returns,advantages)

    if train_epochs % TEST_EPOCHS == 0:
        score = test_agent(env)
        print(score)
        if score >= TARGET_SCORE:
            early_stop = True

    train_epochs += 1


    if i>0 and i % RESTORE_DAMAGE == 0:
        myshelve = shelve.open(filename,'n')
        for key in dir():
            try:
                myshelve[key] = globals()[key]
            except TypeError:
                pass
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        myshelve.close()
        del myshelve
        env.robot.worldReload()
