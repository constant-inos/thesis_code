import numpy as np
from tensorflow.keras.layers import Input,Conv2D,Concatenate,Dense,Flatten
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import os


class ICM:
    # Intrinsic Curiosity Module
    def __init__(self,lr=1e-4,image_shape=(52,39,4),n_actions=3):
        self.lr = lr
        self.image_shape = image_shape
        self.features_dim = 0
        self.n_actions = n_actions

        self.eta = 0.1
        self.beta = 0.2
        self.lamda = 0.1

        self.encoder, self.inverse = self.create_inverse_model()
        self.forward = self.create_forward_model()

        self.load_models()


    def create_inverse_model(self):
        
        state = Input(self.image_shape)
        new_state = Input(self.image_shape)

        conv_1 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(state)
        conv_2 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(conv_1)
        conv_3 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(conv_2)
        conv_4 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(conv_3)
        phi = Flatten()(conv_4)
        self.features_dim = phi.shape[1]

        conv_5 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(new_state)
        conv_6 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(conv_5)
        conv_7 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(conv_6)
        conv_8 = Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(conv_7)
        new_phi = Flatten()(conv_8)

        concat = Concatenate(axis=-1)([phi,new_phi])
        dense1 = Dense(256,activation='relu')(concat)
        dense2 = Dense(self.n_actions,activation='softmax')(dense1)

        inverse = Model(inputs=[state,new_state],outputs=[dense2])
        inverse.compile(optimizer=Adam(lr=self.lr),loss='mse')

        encoder = Model(inputs=new_state,outputs=new_phi)
        #encoder = Model(inputs=state,outputs=inverse.get_layer(new_phi).output)

        return encoder,inverse


    def create_forward_model(self):

        input1 = Input(self.features_dim,)
        input2 = Input(self.n_actions)
        concat = Concatenate(axis=-1)([input1,input2])
        dense1 = Dense(256,activation='relu')(concat)
        dense2 = Dense(self.features_dim,activation='relu')(dense1)

        forward = Model(inputs=[input1,input2],outputs=[dense2])
        forward.compile(optimizer=Adam(self.lr),loss='mse')

        return forward


    def ICM_pass(self,state,action,new_state,train=True):
        state = state.reshape((1,)+state.shape)
        new_state = new_state.reshape((1,)+new_state.shape)
        action = action.reshape((1,)+action.shape)

        at_hat = self.inverse.predict([state,new_state])
        
        phi = self.encoder.predict([state])
        new_phi = self.encoder.predict([new_state])

        new_phi_hat = self.forward.predict([phi,action])

        intrinsic_reward = self.eta/2 * K.mean(K.square(new_phi_hat - new_phi)).numpy()        

        if train:
            inv_hist = self.inverse.fit([state,new_state],[action],verbose=0)
            for_hist = self.forward.fit([phi,action],[new_phi],verbose=0)
            inv_loss = inv_hist.history['loss'][0]
            for_loss= for_hist.history['loss'][0]

            #print(np.float32(inv_loss),np.float32(for_loss))
            #print(at_hat,action)

        return intrinsic_reward,inv_loss,for_loss

    def save_models(self):
        self.forward.save_weights('forward_model.h5')
        self.inverse.save_weights('inverse_model.h5')
        print('model saved')

    def load_models(self):
        if os.path.exists('forward_model.h5'):
            self.forward.load_weights('forward_model.h5')
            self.inverse.load_weights('inverse_model.h5')
            print('ICM model loaded')



