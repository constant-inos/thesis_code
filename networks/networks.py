import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Concatenate,MaxPooling2D


class DenseNet(keras.Model):
    def __init__(self,units=[64]):
        super(DenseNet, self).__init__()

        self.Layers = []
        for n in units:
            self.Layers.append( Dense(units=n, activation='relu') )

    def call(self,INPUT):
        x = INPUT

        for layer in self.Layers:
            x = layer(x)
            
        return x
        
        
class ConvNet(keras.Model):
    def __init__(self,filters=[64,64]):
        super(ConvNet, self).__init__()

        self.Layers = []
        for n in filters:
            self.Layers.append( Conv2D(filters=n, kernel_size=3, activation='relu') )
            self.Layers.append( MaxPooling2D((2,2)) )
        self.Layers.append( Flatten() )

    def call(self,INPUT):
        x = INPUT

        for layer in self.Layers:
            x = layer(x)
            
        return x
        
        
class SimpleDQN(keras.Model):
    def __init__(self,output_size):
        super(SimpleDQN,self).__init__()
        
        self.main = DenseNet(units=[128,128,128])
        self.out = Dense(output_size,activation='linear')
        
    def call(self,INPUT):
        x = INPUT
        x = self.main(x)
        x = self.out(x)
        return x
        
class ConvDQN(keras.Model):
    def __init__(self,output_size):
        super(ConvDQN,self).__init__()
        
        self.conv = ConvNet(filters=[64,64])
        self.main = DenseNet(units=[128,128])
        self.out = Dense(output_size,activation='linear')
        
    def call(self,INPUT):
        x = INPUT

        x = self.conv(x)
        x = self.main(x)
        x = self.out(x)
        
        return x
        
class ComplexDQN(keras.Model):
    def __init__(self,output_size):
        super(Net0,self).__init__()
        
        self.conv = ConvNet(filters=[64,64])
        self.simple = DenseNet(units=[64])
        self.concat = Concatenate(axis=-1)
        
        self.main = DenseNet(units=[128,128])
        self.out = Dense(output_size,activation='linear')
        
    def call(self,INPUT):
        x1 = INPUT[0] # camera data
        x2 = INPUT[1] # sensors data
        
        x1 = self.conv(x1)
        x2 = self.simple(x2)
        x = self.concat([x1,x2])
        x = self.main(x)
        x = self.out(x)
        
        return x
        
        
class SimplePGNet(keras.Model):
    def __init__(self,n_actions):
        super(SimplePGNet, self).__init__()
        self.n_actions = n_actions

        self.main = DenseNet(units=[24,48])
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        x = state
        x = self.main(x)
        pi = self.pi(x)

        return pi

class SimpleACNet(keras.Model):
    def __init__(self, n_actions, name='actor_critic'):
        super(SimpleACNet, self).__init__()
        self.n_actions = n_actions
        self.model_name = name

        self.main = DenseNet([256,512])
        self.v = Dense(1, activation='linear')
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        x = state
        x = self.main(x)
        pi = self.pi(x)
        v = self.v(x)
        
        return v,pi


class SimpleDDPG_actor(keras.Model):
    def __init__(self,n_actions,name='ddpg_actor'):
        super(SimpleDDPG_actor, self).__init__()
        self.n_actions = n_actions
        self.model_name=name 
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # ACTOR

        self.actor_main = DenseNet([256,256])
        self.actor_out = Dense(n_actions,activation='tanh',kernel_initializer=last_init)


    def call(self,state):
        x = self.actor_main(state)
        action = self.actor_out(x)  # UPPER BOUND?
        return action 

class SimpleDDPG_critic(keras.Model):
    def __init__(self,n_actions,name='ddpg_critic'):
        super(SimpleDDPG_critic,self).__init__()
        self.n_actions = n_actions
        self.model_name=name 

        self.critic_state_in = DenseNet([16,32])
        self.critic_action_in = DenseNet([32])

        self.critic_main = DenseNet([256,256])
        self.critic_out = Dense(n_actions)

    def call(self,state,action):
        x_a = self.critic_state_in(state)
        x_b = self.critic_action_in(action)
        x = Concatenate(axis=-1)([x_a,x_b])
        x = self.critic_main(x)
        value = self.critic_out(x)
        return value


