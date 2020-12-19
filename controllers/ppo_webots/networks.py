import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Concatenate

class PPONetwork(keras.Model):
    def __init__(self,n_actions,conv=False):
        super(PPONetworkConv,self).__init__()

        self.HiddenLayers = []

        if conv:
            self.HiddenLayers.append( Conv2D(32,kernel_size=8,strides=(4,4),activation='relu') )
            self.HiddenLayers.append( Conv2D(64,kernel_size=4,strides=(2,2),activation='relu') )
            self.HiddenLayers.append( Conv2D(64,kernel_size=3,activation='relu') )
            self.HiddenLayers.append( Flatten() )

        self.HiddenLayers.append( Dense(256,activation='relu') )
        self.HiddenLayers.append( Dense(256,activation='relu') )
        
        self.v = Dense(1,activation='linear')
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        x = state

        for layer in self.HiddenLayers:
            x = layer(x)

        policy = self.pi(x)
        value = self.v(x)

        return policy, value


# class PPONetwork(keras.Model):
#     def __init__(self,n_actions):
#         super(PPONetwork,self).__init__()

#         self.fc1 = Dense(256,activation='relu')
#         self.fc2 = Dense(256,activation='relu')
        
#         self.v = Dense(1,activation='linear')
#         self.pi = Dense(n_actions,activation='softmax')

#     def call(self,state):
#         x = self.fc1(state)
#         x = self.fc2(x)

#         policy = self.pi(x)
#         value = self.v(x)

#         return policy, value

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
    def __init__(self,action_size,conv=False):
        super(DQNetwork, self).__init__()
        self.HiddenLayers = []

        if conv:
            self.HiddenLayers.append( Conv2D(32,kernel_size=8,strides=(4,4),activation='relu') )
            self.HiddenLayers.append( Conv2D(64,kernel_size=4,strides=(2,2),activation='relu') )
            self.HiddenLayers.append( Conv2D(64,kernel_size=3,activation='relu') )
            self.HiddenLayers.append( Flatten() )
        self.HiddenLayers.append( Dense(units=512, activation='relu') )

        self.value = Dense(units=action_size, activation='linear')

    def call(self,state):
        x = state

        for layer in self.HiddenLayers:
            x = layer(x)

        value = self.value(x)

        return value

class MitsosPPONet(keras.Model):
    def __init__(self,n_actions):
        super(MitsosPPONet, self).__init__()
        self.ConvLayers = []
        self.ConvLayers.append( Conv2D(64,kernel_size=9,activation='relu') )
        self.ConvLayers.append( Conv2D(64,kernel_size=5,activation='relu') )
        self.ConvLayers.append( Conv2D(64,kernel_size=3,activation='relu') )
        
        self.flatten = Flatten() 
        self.concat = Concatenate(axis=-1)
        
        self.DenseLayers = []
        self.DenseLayers.append( Dense(512,activation='relu') )
        self.DenseLayers.append( Dense(256,activation='relu') )

        self.policy = Dense(n_actions,activation='softmax')
        self.value = Dense(1,activation='linear')
    
    def call(self,state):
        x1 = state[0] #stacked frames
        x2 = state[1] #stacked sensor values
        
        for layer in self.ConvLayers:
            x1 = layer(x1)
            
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x = self.concat([x1,x2])
        
        for layer in self.DenseLayers:
            x = layer(x)
        
        pi = self.policy(x)
        v = self.value(x)
        
        return pi,v
        
    