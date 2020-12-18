import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #cut annoying tf messages
from tensorflow.keras.optimizers import Adam
from networks import ActorCriticNetwork
import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp

class Agent(object):
    def __init__(self, n_actions=2,lr=0.01, gamma=0.99):
        self.lr=lr
        self.gamma=gamma
        self.n_actions=n_actions
        self.action_space = [i for i in range(n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(Adam(lr=lr))

        self.action = None

    def choose_action(self,state):
        state = tf.convert_to_tensor([state])
        _, probs = self.actor_critic(state)
        action = np.random.choice(self.action_space,p=probs.numpy()[0])
        return action

    def save_model(self):
        self.actor_critic.save_weights(self.actor_critic.model_name)

    def load_model(self):
        self.actor_critic.load_weights(self.actor_critic.model_name)

    def learn(self,state,action,reward,state_,done):

        state = tf.convert_to_tensor([state])
        state_ = tf.convert_to_tensor([state_])
        #action = tf.convert_to_tensor([action])
        reward = tf.convert_to_tensor([reward])

        with tf.GradientTape() as tape:
            value, probs = self.actor_critic(state)
            value_, probs_ = self.actor_critic(state_)
            value = tf.squeeze(value)
            value_ = tf.squeeze(value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(tf.convert_to_tensor(action))

            """
            log_prob = -sparse_categorical_crossentropy_with_logits
            what is the loss function exactly ??? calculate it 
            (how tf works, sess, graph, fast?)
            """

            delta = reward + self.gamma * value_ * (1-int(done)) - value 
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient,self.actor_critic.trainable_variables))



if __name__ == '__main__':
    import gym

    env = gym.make('CartPole-v0')
    agent = Agent(lr= 0.9*1e-5,n_actions=env.action_space.n)
    n_games = 2000

    score_history = []
    max_score, max_avg = 0,0


    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.choose_action(obs)
            obs_,reward,done,info = env.step(action)
            score += reward
            agent.learn(obs,action,reward,obs_,done)
            obs = obs_
            steps += 1
        score_history.append(score)
        avg_score  = np.mean(score_history[-100:])

        print('GAMES:',i,'SCORE:',score,'AVG SCORE:',avg_score)
        if i % 100 == 0: print(max_score,max_avg)
        if score > max_score: max_score = score
        if avg_score > max_avg: max_avg = avg_score
