import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv2D,Flatten


class PPONetwork(keras.Model):
    def __init__(self,n_actions):
        super(PPONetwork,self).__init__()

        self.fc1 = Dense(256,activation='relu')
        self.fc2 = Dense(256,activation='relu')
        
        self.v = Dense(1,activation='linear')
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        x = self.fc1(state)
        x = self.fc2(x)

        policy = self.pi(x)
        value = self.v(x)

        return policy, value

class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, name='actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.model_name = name

        self.layer1 = Dense(1024, activation='relu')
        self.layer2 = Dense(512, activation='relu')
        self.v = Dense(1, activation='linear')
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        value = self.layer1(state)
        value = self.layer2(value)

        pi = self.pi(value)
        v = self.v(value)
        
        return v,pi

class PolicyGradientNetwork(keras.Model):
    def __init__(self,n_actions):
        super(PolicyGradientNetwork, self).__init__()
        self.n_actions = n_actions

        self.fc1 = Dense(256,activation='relu')
        self.fc2 = Dense(256,activation='relu')
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)

        return pi


class DQNetwork(keras.Model):
    def __init__(self,action_size):
        super(DQNetwork, self).__init__()
        self.HiddenLayers = []

        # self.HiddenLayers.append( Conv2D(32,kernel_size=8,strides=(4,4),activation='relu') )
        # self.HiddenLayers.append( Conv2D(64,kernel_size=4,strides=(2,2),activation='relu') )
        # self.HiddenLayers.append( Conv2D(64,kernel_size=3,activation='relu') )
        # self.HiddenLayers.append( Flatten() )
        self.HiddenLayers.append( Dense(units=512, activation='relu') )

        self.value = Dense(units=action_size, activation='linear')

    def call(self,state):
        x = state

        for layer in self.HiddenLayers:
            x = layer(x)

        value = self.value(x)

        return value

