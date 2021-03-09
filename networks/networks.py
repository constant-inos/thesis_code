import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Concatenate,MaxPooling2D

class PPONetwork(keras.Model):
    def __init__(self,n_actions,conv=False):
        super(PPONetwork,self).__init__()

        self.HiddenLayers = []

        if conv:
            self.HiddenLayers.append( Conv2D(32,kernel_size=8,strides=(4,4),activation='relu') )
            self.HiddenLayers.append( Conv2D(64,kernel_size=4,strides=(2,2),activation='relu') )
            self.HiddenLayers.append( Conv2D(64,kernel_size=3,activation='relu') )
            self.HiddenLayers.append( Flatten() )

        self.HiddenLayers.append( Dense(256,activation='relu') )
        self.HiddenLayers.append( Dense(512,activation='relu') )
        
        self.v = Dense(1,activation='linear')
        self.pi = Dense(n_actions,activation='softmax')

    def call(self,state):
        x = state

        for layer in self.HiddenLayers:
            x = layer(x)

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
        

class MitsosDQNet(keras.Model):
    def __init__(self,action_size):
        super(MitsosDQNet, self).__init__()
        self.ConvLayers = []
        self.ConvLayers.append( Conv2D(64,kernel_size=9,activation='relu') )
        self.ConvLayers.append( Conv2D(64,kernel_size=5,activation='relu') )
        #self.ConvLayers.append( Conv2D(64,kernel_size=3,activation='relu') )
        
        self.flatten = Flatten() 
        self.concat = Concatenate(axis=-1)
        
        self.DenseLayers = []
        self.DenseLayers.append( Dense(units=512, activation='relu') )
        self.DenseLayers.append( Dense(units=512, activation='relu') )
        self.DenseLayers.append( Dense(units=512, activation='relu') )

        self.value = Dense(units=action_size, activation='linear')
    
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
        
        v = self.value(x)
        
        return v


################################################################################################################################

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
        super(Net1,self).__init__()
        
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
