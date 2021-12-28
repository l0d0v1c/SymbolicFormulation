import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
import numpy as np
import random
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, Bidirectional,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential  
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras import backend as K
import textdistance
from statistics import mean 
import pickle
import pandas as pd


class KLDivergenceLayer(Layer):

    

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def nll(y_true, y_pred):
            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    
class AE:
    def __init__(self,name="out"):
        self.name=name
        self.mode='VAE'
        self.PAD_CHAR = "~"
        np.random.seed(7)
        random.seed(77)
        print('''____ ___    _  _ ____ ___  _ ____ ___ _ ____ __ _
 |--<  |__>   |\/| |=== |__> | |--|  |  | [__] | \|''')
        if(tf.__version__[0]=='1'):
            tf.random.set_random_seed(5367)
        else:
            tf.random.set_seed(5367)
        print("Tensorflow version %s" %(tf.__version__,))
        print("GPU %s" %(tf.test.is_gpu_available(),))
        
    def OneHot(self,df):
        self.ST_LENGTH = max([len(p) for p in df])
        self.tmpdf=df
        padded_st = []
        self.charset = set(self.PAD_CHAR)
        for p in df:
            padded_st.append(p.ljust(self.ST_LENGTH, self.PAD_CHAR))
            self.charset |= set(p)
        self.charset=list(self.charset)
        self.vocab_size = len(self.charset)
        self.char2id = dict((c, i) for i, c in enumerate(self.charset))
        
        self.encoded_st = [[self.char2id[c] for c in vl] for vl in padded_st]
        self.one_hot_encoded = np.array([to_categorical(p, num_classes=self.vocab_size) for p in self.encoded_st])
        
    
    def reload(self,name):
        self.name=name
        try:
            self.encoder=load_model('enc%s.h5'%(self.name,))
            self.decoder=load_model('decl%s.h5'%(self.name,))
            self.autoencoder=load_model('model%s.h5'%(self.name,))
            self.mode='LSTM'
        except:
            self.encoder=load_model('enc%s.h5'%(self.name,),custom_objects={'KLDivergenceLayer':KLDivergenceLayer,'nll':nll})
            self.decoder=load_model('decl%s.h5'%(self.name,),custom_objects={'KLDivergenceLayer':KLDivergenceLayer,'nll':nll})
            #self.autoencoder=load_model('model%s.h5'%(self.name,),custom_objects={'KLDivergenceLayer':KLDivergenceLayer,'nll':nll})
            self.mode='VAE'
            
        tmp=pickle.load(open('dico%s.pkl'%(self.name,),'rb'))
        self.vocab_size=tmp[0]
        self.char2id=tmp[1]
        self.charset=tmp[2]
        self.latentsize=tmp[3]
        self.ST_LENGTH=tmp[4]

    def generate(self):
        st_str=self.decode(np.array([[random.uniform(-1, 1) for _ in range(self.latentsize)]]))
        return st_str
    def encode(self,formule):
        padded_st=[formule.ljust(self.ST_LENGTH, self.PAD_CHAR)]
        encoded_st = [[self.char2id[c] for c in vl] for vl in padded_st]
        one_hot_encoded = np.array([to_categorical(p, num_classes=self.vocab_size) for p in encoded_st])
        if self.mode=='LSTM':
            v=self.encoder.predict(one_hot_encoded)
        else:
            v=self.encoder.predict(one_hot_encoded.reshape(-1,self.ST_LENGTH*self.vocab_size))
        return v
    def decode(self, latent_sample):
        unpad = lambda text: text.replace(self.PAD_CHAR, "")
        one_hot_decode = lambda one_hot_vectors: "".join([list(self.charset)[np.argmax(vec)] for vec in one_hot_vectors])
        st_vec = self.decoder.predict(latent_sample)
        if self.mode=='LSTM':
            st_str = unpad(one_hot_decode(st_vec[0]))
        else:
            st_str = unpad(one_hot_decode(st_vec[0].reshape(self.ST_LENGTH,self.vocab_size)))
        return st_str
    def analyse(self):
        unpad = lambda text: text.replace(self.PAD_CHAR, "")
        one_hot_decode = lambda one_hot_vectors: "".join([list(self.charset)[np.argmax(vec)] for vec in one_hot_vectors])
        
        if self.mode=="LSTM":
            reconst_vecs = self.autoencoder.predict(self.one_hot_encoded)
            reconst_str = [unpad(one_hot_decode(p)) for p in reconst_vecs]
            self.compare = pd.DataFrame(zip(self.tmpdf, reconst_str),columns = ['Original', 'Reconstructed'])
        else:
            encoded=self.encoder.predict(self.savex_t)
            decoded=self.decoder.predict(encoded).reshape(-1,self.ST_LENGTH,self.vocab_size)
            reconst = [unpad(one_hot_decode(p)) for p in decoded]
            self.compare=pd.DataFrame([[i,j] for i,j in zip(self.tmpdf,reconst)], columns = ['Original', 'Reconstructed'])
            
        
        RO=mean([textdistance.ratcliff_obershelp(j[0],j[1]) for i,j in self.compare.iterrows()])
        RS=mean([textdistance.sorensen(j[0],j[1]) for i,j in self.compare.iterrows()])
        print("Ratcliff-Obershelp=%.2f Sorensen=%.2f" % (RO,RS))
    
    
