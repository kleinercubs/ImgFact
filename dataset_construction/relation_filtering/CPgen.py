#_*_coding:utf-8_*_
import os
os.environ['TF_KERAS'] = "1"
import sys
import json
import random
import numpy as np
from bert4keras.backend import keras, K
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from bert4keras.snippets import DataGenerator,sequence_padding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.layers import Loss
from mlayer import Slice
import tensorflow as tf
from functools import partial
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='001', help='File index.')

args = parser.parse_args()
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
  

AUTOTUNE = tf.data.AUTOTUNE
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR,"data")
MODEL_DIR = os.path.join(ROOT_DIR,"models")

N = 15
max_text = 20

config_path = os.path.join(MODEL_DIR,"bert","bert_config.json")
checkpoint_path = os.path.join(MODEL_DIR,"bert","bert_model.ckpt")
dict_path = os.path.join(MODEL_DIR,"bert","vocab.txt")

tokenizer = Tokenizer(dict_path,do_lower_case=True)

def read_tfrecord(example):
    example = tf.io.parse_single_example(example, features = {
        'pos_sent': tf.io.FixedLenFeature([], tf.string),
        'neg_sents': tf.io.FixedLenFeature([], tf.string),
        'pair': tf.io.FixedLenFeature([], tf.string),
        'relation': tf.io.FixedLenFeature([], tf.string),
        'so': tf.io.FixedLenFeature([], tf.string),
        'fileno': tf.io.FixedLenFeature([], tf.string),
    })
    return example['pos_sent'], example['neg_sents'], example['pair'], example['relation'], example['so'], example['fileno']

def load_dataset(filename):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filename)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset

def get_dataset(filenames, batch_size):
    dataset = load_dataset(filenames)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def load_data(fn,shuffle=True):
    D = []
    with open(fn,"r",encoding="utf-8") as rf:
        for line in rf:
            line = json.loads(line)
            D.append([line["pair"],line["pos_sent"],line["neg_sents"][:N]])
    print(len(D))
    if shuffle:
        random.shuffle(D)
    return D

class DG(DataGenerator):
    def __init__(self,data,batch_size,**kwargs):
        super(DG,self).__init__(data,batch_size,**kwargs)

    def __iter__(self,random=False):
        X1,X2,X3,X4,X5,X6 = [],[],[],[],[],[]
        for is_end,(pair,pos_sent,neg_sents) in self.sample(random=random):
            compound_noun = pair
            ## compound noun encoding
            token_ids,segment_ids = tokenizer.encode(compound_noun,maxlen=max_text)
            X1.append(token_ids)
            X2.append(segment_ids)
            ## postive sentence encoding
            s_token_ids,s_segment_ids = tokenizer.encode(pos_sent,maxlen=max_text)
            X3.append(s_token_ids)
            X4.append(s_segment_ids)

            ## negative sentence encoding
            text_list = []
            segment_list = []
            for neg_sent in neg_sents[:N]:
                n_token_ids,n_segment_ids = tokenizer.encode(neg_sent,maxlen=max_text)
                n_token_ids = n_token_ids + [0] * (max_text-len(n_token_ids))
                n_segment_ids = n_segment_ids + [0] *  (max_text-len(n_segment_ids))
                text_list += n_token_ids
                segment_list += n_segment_ids
            if len(neg_sents) < N:
                for _ in range(N-len(neg_sents)):
                    text_list += [101,102] + [0] * (max_text-2)
                    segment_list += [0] * max_text
            X5.append(text_list)
            X6.append(segment_list)
            
            if is_end or len(X1) == self.batch_size:
                X1 = sequence_padding(X1)
                X2 = sequence_padding(X2)
                X5 = sequence_padding(X5)
                X6 = sequence_padding(X6)
                X3 = sequence_padding(X3,length=max_text)
                X4 = sequence_padding(X4,length=max_text)
                yield [X1,X2,X3,X4,X5,X6],None
                X1,X2,X3,X4,X5,X6 = [],[],[],[],[],[]

class CPLoss(Loss):
    def __init__(self,temperature=0.1,cosine_sim=True,output_axis=1,**kwargs):
        self.temperature = temperature
        self.cosine_sim = cosine_sim
        super(CPLoss,self).__init__(output_axis,**kwargs)

    def compute_cosine_similarity(self,x1,x2):
        assert len(K.int_shape(x1)) == 2
        assert len(K.int_shape(x2)) == 2 or len(K.int_shape(x2)) == 3
        
        d2 = K.batch_dot(x1,x1)
        if len(K.int_shape(x2)) == 2:
            d1 = K.batch_dot(x1,x2)
            d3 = K.batch_dot(x2,x2)
        else:
            d1 = K.batch_dot(x1,K.permute_dimensions(x2,(0,2,1)))
            s_1,s_2 = K.int_shape(x2)[1:]
            tmp = K.reshape(x2,(-1,s_2))
            d3 = K.batch_dot(tmp,tmp)
            d3 = K.reshape(d3,(-1,s_1))
        denominator = K.maximum(K.sqrt(d2 * d3),K.epsilon())
        simliarity = d1/denominator
        return simliarity

    def compute_loss(self,inputs,mask=None):
        x,pos_x,neg_xx = inputs
        if not self.cosine_sim:
            x1 = K.batch_dot(x,pos_x)/self.temperature
            x2 = K.batch_dot(x,K.permute_dimensions(neg_xx,(0,2,1)))/self.temperature
        else:
            x1 = self.compute_cosine_similarity(x,pos_x)/self.temperature
            x2 = self.compute_cosine_similarity(x,neg_xx)/self.temperature
        tmp = K.concatenate([x1,x2],axis=-1)
        max_val = K.max(tmp,axis=1,keepdims=True)
        x1 = K.exp(x1-max_val)
        x2 = K.exp(x2-max_val)

        x1 = K.squeeze(x1,1)
        x2 = K.sum(x2,axis=-1)
        x3 = x1 + x2
        loss = K.mean(-K.log(x1/x3 + K.epsilon()))
        return loss

class SaveBest(keras.callbacks.Callback):
    def __init__(self,models,mfns,fn=None,**kwargs):
        self.pair_mlm = models[0]
        self.sent_mlm = models[1]
        self.pair_mfn = mfns[0]
        self.sent_mfn = mfns[1]
        self.val_loss = 100.
        data = []
        fn = os.path.join(DATA_DIR,"explanation/valid_imgfact.json")
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                data.append([d["pair"],d["pos_sent"],d["neg_sents"]])
        self.data = data
        super(SaveBest,self).__init__(**kwargs)

    def cal_accuracy(self,):
        acc_count = 0
        for d in self.data:
            token_ids,segment_ids = tokenizer.encode(d[0])
            pair_cls = self.pair_mlm.predict([np.array([token_ids]),np.array([segment_ids])])[0,0,:]
            cur_res = []
            candidates = [d[1]] + d[2]
            for sent in candidates:
                sent_token,sent_segment = tokenizer.encode(sent)
                sent_cls = self.sent_mlm.predict([np.array([sent_token]),np.array([sent_segment])])[0,0,:]
                cur_res.append(np.dot(pair_cls,sent_cls))
            cur_res = np.array(cur_res)
            y_pred = np.argmax(-cur_res)[:5]
            if 0 in y_pred[:5]:
                acc_count += 1
        return acc_count/len(self.data)

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get("val_loss")
        print("Epoch",epoch)#,": Accuracy: %f" % self.cal_accuracy())
        if val_loss < self.val_loss:
            self.val_loss = val_loss
            self.pair_mlm.save_weights(self.pair_mfn)
            self.sent_mlm.save_weights(self.sent_mfn)

class CPModel:
    def __init__(self,epochs,batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.pair_MLM = None
        self.sent_MLM = None

    def build(self,compile=True):
        in_1 = keras.layers.Input(shape=(None,))
        in_2 = keras.layers.Input(shape=(None,))
        in_3 = keras.layers.Input(shape=(max_text,))
        in_4 = keras.layers.Input(shape=(max_text,))

        in_5 = keras.layers.Input(shape=(max_text * N,))
        in_6 = keras.layers.Input(shape=(max_text * N,))

        self.pair_MLM = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path)
        self.sent_MLM = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path)
        CLS = keras.layers.Lambda(lambda x:x[:,0])
        TEXT_SLICE = Slice(N,max_text)
        
        NN_x = self.pair_MLM([in_1,in_2])
        NN_x = CLS(NN_x)
        POS_x = self.sent_MLM([in_3,in_4])
        POS_x = CLS(POS_x)

        in_5_slice  = TEXT_SLICE(in_5)
        in_6_slice  = TEXT_SLICE(in_6)
        mlm_outs = []
        for i in range(N):
            out = self.sent_MLM([in_5_slice[i],in_6_slice[i]])
            out = CLS(out)
            mlm_outs.append(out)
        NEG_x = keras.layers.Concatenate(axis=-1)(mlm_outs)
        NEG_x = keras.layers.Reshape(target_shape=(N,-1))(NEG_x)
        loss = CPLoss(0.15,True,1)([NN_x,POS_x,NEG_x])
        self.model = keras.models.Model(inputs=[in_1,in_2,in_3,in_4,in_5,in_6],outputs=loss)
        self.model.summary()
        if compile:
            self.model.compile(Adam(1e-5))

    def train(self,train_data,valid_data,mfns=None,eval_fn=None,plot=True):
        
        # strategy = tf.distribute.MultiWorkerMirroredStrategy()  # 建立单机多卡策略
        # with strategy.scope():  # 调用该策略
        if not self.model:
            self.build(compile=True)
        if mfns:
            self.pair_MLM.load_weights(mfns[0])
            self.sent_MLM.load_weights(mfns[1])
        
        train_D = DG(train_data,self.batch_size)
        valid_D = DG(valid_data,self.batch_size)
        save_best = SaveBest([self.pair_MLM,self.sent_MLM],[os.path.join(MODEL_DIR,"model_pair_v0428.h5"),os.path.join(MODEL_DIR,"model_sent_v0428.h5")])
        callbacks = [save_best]
        history = self.model.fit_generator(
            train_D.forfit(),
            steps_per_epoch=len(train_D),
            epochs=self.epochs,
            validation_data=valid_D.forfit(),
            validation_steps=len(valid_D),
            callbacks=callbacks,
        )
        if plot:
            # print(history.history.keys())
            try:
                self.plot_results(history)
            except Exception as e:
                print(e)

    def inference(self,fn,out_fn,mfns=None):
        if not self.model:
            self.build(compile=False)
        if mfns:
            self.pair_MLM.load_weights(mfns[0])
            self.sent_MLM.load_weights(mfns[1])
    
        data = []
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                data.append([d["pair"],d["pos_sent"],d["neg_sents"],d["relation"]])
        res = []
        acc_count = 0
        tot_count = 0
        bar = tqdm(data)
        for d in bar:
            token_ids,segment_ids = tokenizer.encode(d[0])
            pair_cls = self.pair_MLM.predict([np.array([token_ids]),np.array([segment_ids])])[0,0,:]
            cur_res = []
            candidates = [d[1]] + d[2]
            for sent in candidates:
                sent_token,sent_segment = tokenizer.encode(sent)
                sent_cls = self.sent_MLM.predict([np.array([sent_token]),np.array([sent_segment])])[0,0,:]
                # print(pair_cls.shape, sent_cls.shape)
                cur_res.append(np.dot(pair_cls,sent_cls))
            cur_res = np.array(cur_res)
            y_pred = np.argsort(cur_res)[:5]
            tot_count += 1
            if 0 in y_pred:
                acc_count += 1
            x = d + [[candidates[i] for i in y_pred]] + [[str(i) for i in y_pred]] + [[str(i) for i in cur_res]]
            res.append(x)
            bar.set_description("hit@10: {}, ACC@10: {}".format(acc_count, acc_count/tot_count))
        with open(out_fn,"w",encoding="utf-8") as wf:
            for x in res:
                d = {
                    "pair":x[0],
                    "pos_sent":x[1],
                    "neg_sents":x[2],
                    "relation":x[3],
                    "pred_sent":x[4],
                    "pred_index":x[5],
                    "pred_val":x[6]
                }
                # print(d)
                wf.write(json.dumps(d,ensure_ascii=False)+"\n")

    def infer_out(self,fn,out_fn,mfns=None):
        if not self.model:
            self.build(compile=False)
        if mfns:
            self.pair_MLM.load_weights(mfns[0])
            self.sent_MLM.load_weights(mfns[1])
        
        wf = open(out_fn,"w",encoding="utf-8")
        dataset = get_dataset(fn, 32)
        
        acc_count = 0
        tot_count = 0
        bar = tqdm(iter(dataset))
        for pos_sent, neg_sents, pair, relation, so, fileno in bar: 
            token_ids_list, segment_ids_list = [], []
            for x in pair:
                token_ids,segment_ids = tokenizer.encode(str(x))
                token_ids = (token_ids + [0] * 64)[:64]
                segment_ids = (segment_ids + [0] * 64)[:64]
                token_ids_list.append(token_ids)
                segment_ids_list.append(segment_ids)
            pair_cls = self.pair_MLM([np.array(token_ids_list),np.array(segment_ids_list)], training=False)
            pair_cls = pair_cls[:,0,:]
            for pair, neg, pos, rel, spo, no in zip(pair_cls, neg_sents, pos_sent, relation, so, fileno):
                candidates = [str(pos)] + str(neg).split('__')
                token_ids_list, segment_ids_list = [], []
                cur_res = []
                for sent in candidates:
                    sent_token,sent_segment = tokenizer.encode(sent)
                    sent_token = (sent_token + [0] * 64)[:64]
                    sent_segment = (sent_segment + [0] * 64)[:64]
                    token_ids_list.append(sent_token)
                    segment_ids_list.append(sent_segment)
                sent_cls = self.sent_MLM([np.array(token_ids_list),np.array(segment_ids_list)], training=False)[:,0,:]
                for sent in sent_cls:
                    cur_res.append(np.dot(pair,sent))
                cur_res = np.array(cur_res)
                y_pred = np.argsort(-cur_res)[:5]
                tot_count += 1
                if 0 in y_pred:
                    acc_count += 1
                    wf.writelines('{}\t{}\t{}\t{}\n'.format(
                        rel.numpy().decode('utf-8'), 
                        spo.numpy().decode('utf-8').split('##')[0], 
                        spo.numpy().decode('utf-8').split('##')[1], 
                        no.numpy().decode('utf-8'))
                    )
                    wf.flush()
            bar.set_description("capture: {}, ACC@5: {}".format(acc_count, acc_count/tot_count))
            
    def evaluate(self,fn,show_error=True):
        acc = 0
        count = 0
        res = []
        with open(fn,"r",encoding="utf-8") as rf:
            for line in rf:
                d = json.loads(line)
                if d["pos_sent"] == d["pred_sent"][0]:
                    acc += 1
                else:
                    res.append(d)
                count += 1
        # print("Accuracy: %f" % (acc/count))
        if show_error:
            _d = os.path.dirname(fn)
            _f = os.path.basename(fn).replace(".","_error.")
            error_fn = os.path.join(_d,_f)
            with open(error_fn,"w",encoding="utf-8") as wf:
                for x in res:
                    wf.write(json.dumps(x,ensure_ascii=False)+"\n")
                    
    def plot_results(self,history):
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.legend(["train","valid"],loc="upper left")
        plt.title("Model Loss");plt.xlabel("epoch");plt.ylabel("loss")
        plt.savefig(os.path.join(DATA_DIR,"imgs","CP_loss.pdf"))
        plt.close()

def main():
    train_fn = os.path.join(DATA_DIR,"explanation/train_imgfact.json")
    valid_fn = os.path.join(DATA_DIR,"explanation/valid_imgfact.json")
    train_data = load_data(train_fn)
    valid_data = load_data(valid_fn)

    model = CPModel(0, 32)
    model.train(train_data,valid_data)
    # model = CPModel(0, 32)
    model.sent_MLM.load_weights(os.path.join(MODEL_DIR,"model_sent_v0428.h5"))
    model.pair_MLM.load_weights(os.path.join(MODEL_DIR,"model_pair_v0428.h5"))

    test_fn = os.path.join(DATA_DIR,"explanation/valid_imgfact.json")
    out_fn = os.path.join(DATA_DIR,"explanation/valid_imgfact_res.json")
    model.inference(test_fn,out_fn)
    # model.evaluate(out_fn)

    # test_fn = os.path.join(DATA_DIR,"title/Triplelist{}.tfrecords".format(args.file))
    # out_fn = os.path.join(DATA_DIR,"title/Triplelist{}.txt".format(args.file))
    # test_fn = os.path.join(DATA_DIR,"explanation/valid_db_new.tfrecords")
    # out_fn = os.path.join(DATA_DIR,"explanation/valid_db_new.txt")
    # model.infer_out(test_fn,out_fn)
    # model.evaluate(out_fn)

if __name__ == '__main__':
    
    main()
