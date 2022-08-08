#_*_coding:utf-8_*_
# import keras
import bert4keras
import numpy as np
import tensorflow as tf
# import keras.backend.tensorflow_backend as K

from bert4keras.backend import keras, K

def seq_gather(inputs):
    x,head,tail = inputs
    head = K.cast(head,"int32")
    tail = K.cast(tail,"int32")
    batch_idx = K.arange(0,K.shape(x)[0])
    batch_idx = K.expand_dims(batch_idx,1)
    head_new = K.concatenate([batch_idx,head],axis=-1)
    tail_new = K.concatenate([batch_idx,tail],axis=-1)
    head_f = tf.gather_nd(x,head_new)
    tail_f = tf.gather_nd(x,tail_new)
    outputs = K.concatenate([head_f,tail_f],axis=-1)
    return outputs

def info_attention(inputs,f_cnt,dim):
    a = keras.layers.Dense(f_cnt*dim,activation="softmax")(inputs)
    a = keras.layers.Reshape((f_cnt,dim))(a)
    a = keras.layers.Lambda(lambda x: K.sum(x,axis=2))(a)
    a = keras.layers.RepeatVector(dim)(a)
    a_probs = keras.layers.Permute((2,1),name="info_atten_vec")(a)
    a_probs = keras.layers.Flatten()(a_probs)
    outputs = keras.layers.Multiply()([inputs,a_probs])
    return outputs

class LossCheck(keras.callbacks.Callback):
    def __init__(self,help_model,valid_D,loss_data_D,loss_diff=0.7,**kwargs):
        self.help_model = help_model
        self.loss_data_D = loss_data_D
        self.valid_D = valid_D
        self.loss_diff = loss_diff
        self.last_loss = -1.
        self.last_pred_res = None
        self.last_pred_loss = None

    def on_epoch_end(self, epoch, logs={}):
        cur_loss = logs.get("val_loss")
        print(epoch,cur_loss)
        if self.last_loss == -1.:
            self.last_loss = cur_loss
            self.last_pred_res = self.model.predict_generator(self.valid_D.__iter__(random=False),steps=len(self.valid_D))
            self.last_pred_loss = self.help_model.predict_generator(self.loss_data_D.__iter__(random=False),steps=len(self.loss_data_D))
            return
        pred_res = self.model.predict_generator(self.valid_D.__iter__(random=False),steps=len(self.valid_D))
        pred_loss = self.help_model.predict_generator(self.loss_data_D.__iter__(random=False),steps=len(self.loss_data_D))
        loss_diff = cur_loss - self.last_loss
        if cur_loss > 1.5 or loss_diff >= self.loss_diff:
            print("Find loss exception point!")
            print(pred_res.shape)
            print(pred_loss.shape)
            np.save("last_loss.npy",self.last_pred_loss)
            np.save("last_res.npy",self.last_pred_res)
            np.save("loss.npy",pred_loss)
            np.save("res.npy",pred_res)
            exit()
        self.last_loss = cur_loss
        self.last_pred_loss = pred_loss
        self.last_pred_res = pred_res

class BaseAttention(keras.layers.Layer):
    def __init__(self,dim,**kwargs):
        self.dim = dim
        # self.count = count
        super(BaseAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.w = self.add_weight(name="{}_W".format(self.name),
                                shape=(self.dim,),
                                dtype="float32",
                                initializer="glorot_uniform",
                                trainable=True
                            )
        super(BaseAttention,self).build(input_shape)

    def call(self,inputs):
        inputs = K.cast(inputs,"float32")
        # inputs = K.reshape(inputs,shape=(-1,self.count,self.dim))
        x = tf.multiply(inputs,self.w)
        x = K.sum(x,axis=-1)
        x = K.softmax(x,axis=-1)
        x = K.repeat(x,self.dim)
        probs = K.permute_dimensions(x,(0,2,1))
        outputs = tf.multiply(inputs,probs)
        return outputs
    
    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[1],input_shape[2]])

class SOAttention(keras.layers.Layer):
    def __init__(self,d1,d2,bias=False,**kwargs):
        self.d1 = d1
        self.d2 = d2
        self.bias = bias
        super(SOAttention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W = self.add_weight("{}_W".format(self.name),
                                shape=(self.d1,self.d2),
                                initializer="glorot_uniform",
                                trainable=True
                            )
        if self.bias:
            self.b = self.add_weight("{}_b".format(self.name),
                                shape=(self.d2,),
                                initializer="glorot_uniform",
                                trainable=True
                            )
                
    def call(self,inputs):
        q,k = inputs
        x = K.dot(q,self.W)
        x = K.batch_dot(x,k)
        if self.bias:
            x += self.b
        x = K.softmax(K.cast(x,"float32"),axis=-1)
        # zeros = K.zeros_like(x)
        # x = tf.where(x > 0.2, x, zeros)
        probs = K.permute_dimensions(K.repeat(x,self.d1),(0,2,1))
        outputs = tf.multiply(q,probs)
        return outputs
        
    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[0][1],input_shape[0][2]])

class Slice(keras.layers.Layer):
    def __init__(self,bag_num,dim,**kwargs):
        self.dim = dim
        self.bag_num = bag_num
        super(Slice,self).__init__(**kwargs)

    def slice(self,x,index):
        return tf.slice(x,[0,index * self.dim],[-1,self.dim])

    def call(self,inputs):
        outputs = []
        for i in range(self.bag_num):
            output = self.slice(inputs,i)
            outputs.append(output)
        return outputs

    def compute_output_shape(self,input_shape):
        return [tuple([None,self.dim])] * self.bag_num

class TypeEmbedding(keras.layers.Layer):
    def __init__(self,type_num,type_dim,**kwargs):
        self.type_num = type_num 
        self.type_dim = type_dim
        self.initializer = "glorot_uniform"
        super(TypeEmbedding,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.type_matrix = self.add_weight(name="{}_W".format(self.name),
                                    shape=[self.type_num,self.type_dim],
                                    initializer=self.initializer,
                                    trainable=True
                                )
        super(TypeEmbedding,self).build(input_shape)

    def call(self,inputs):
        inputs = K.cast(inputs,"int32")
        outputs = tf.gather(self.type_matrix,inputs,axis=0)
        print(type(outputs))
        outputs = tf.reshape(outputs,[-1,2 * self.type_dim])
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([input_shape[0],2 * self.type_dim])

class GateMask(keras.layers.Layer):
    def __init__(self,dim,**kwargs):
        self.dim = dim
        # self.activation = keras.activations.get(activation,None)
        super(GateMask,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W = self.add_weight(name="{}_W".format(self.name),
                                shape=(self.dim,self.dim),
                                dtype="float32",
                                initializer="glorot_uniform",
                                trainable=True
                            )
        super(GateMask,self).build(input_shape)

    def call(self,inputs):
        x = K.dot(inputs,self.W)
        g = K.sigmoid(x)
        outputs = tf.multiply(inputs,g)
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[1]])

class RGCNLayer(keras.layers.Layer):
    def __init__(self,in_feat,out_feat,num_rels,num_bases=-1,bias=None,activation=None,adj_node=None,**kwargs):
        super(RGCNLayer,self).__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.activation = keras.activations.get(activation)
        self.bias = bias
        self.adj_node = adj_node
        # self.is_input_layer = is_input_layer

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        

    def build(self,input_shape):
        self.weight = self.add_weight("{}_W".format(self.name),
                                    shape=(self.num_bases,self.in_feat,self.out_feat),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        self.Wo = self.add_weight("{}_W0".format(self.name),
                                    shape=(self.in_feat,self.out_feat),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        if self.num_bases < self.num_rels:
            self.w_cmp = self.add_weight("{}_cmp".format(self.name),
                                    shape=(self.num_rels,self.num_bases),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )
        if self.bias:
            self.b = self.add_weight("{}_bias".format(self.name),
                                    shape=(self.out_feat,),
                                    initializer="glorot_uniform",
                                    trainable=True
                                )

    def call(self,inputs):
        if self.num_bases < self.num_rels:
            self.weight = K.reshape(self.weight,(self.num_bases,self.in_feat,self.out_feat))
            self.weight = K.permute_dimensions(self.weight,(1,0,2))
            weight = K.dot(self.w_cmp,self.weight)
            weight = K.reshape(weight,(self.num_rels,self.in_feat,self.out_feat))
        else:
            weight = self.weight
        
        features = inputs[0]
        A = inputs[1:]
        xs = []
        for i in range(self.num_rels):
            edges = A[i]
            x = K.dot(features,weight[i])
            x = K.batch_dot(K.permute_dimensions(edges,(0,2,1)),x) / (K.sum(edges,axis=2,keepdims=True) + K.epsilon())
            xs.append(x)
        eyes_matrix = K.eye(self.adj_node)
        eyes_matrix = K.repeat_elements(eyes_matrix,K.shape(inputs[0])[0],axis=0)
        xs.append(K.batch_dot(eyes_matrix,K.dot(features,self.Wo)))
        outputs = K.concatenate(xs,axis=-1)
        outputs = K.reshape(outputs,(-1,self.adj_node,self.out_feat,self.num_rels+1))
        outputs = K.sum(outputs,axis=-1)
        if self.bias:
            outputs += self.b
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self,input_shape):
        return tuple([None,input_shape[0][0],self.out_feat])

class GlobalGCN(keras.layers.Layer):
    def __init__(self,node_num,dim,M,**kwargs):
        self.M = M
        self.node_num = node_num
        self.dim = dim
        super(GlobalGCN,self).__init__(**kwargs)

    def build(self,input_shape):
        super(GlobalGCN,self).build(input_shape)

    def call(self,inputs):
        pass

    def compute_output_shape(self,input_shape):
        pass