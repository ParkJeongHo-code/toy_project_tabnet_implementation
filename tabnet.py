
import tensorflow_addons as tfa
from sklearn.metrics import balanced_accuracy_score,accuracy_score
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import optimizers
from collections import Counter

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
    print(e)

import shap

shap.initjs()
X,y = shap.datasets.adult()
X_display,y_display = shap.datasets.adult(display=True)
from sklearn.model_selection import train_test_split
normalization_df = (X - X.min())/(X.max()-X.min())

train_x, test_x, train_lbls, test_lbls = train_test_split(normalization_df, y, test_size=0.2, random_state=7)


class tabnet_classifier():
    def __init__(self,input_f_n,last_mlp_n, ds_node_num,step_num,class_num,ramda=1.3):
        self.ds_node_num=ds_node_num
        self.step_num=step_num
        self.ramda=ramda
        self.prior_scale=tf.ones([1,input_f_n])
        self.last_mlp_n=last_mlp_n
        self.class_num=class_num

    def shared_decision(self,inputs,middle_output):
        inputs=tf.keras.layers.Input(shape=inputs)
        x=tf.keras.layers.Dense(middle_output)(inputs)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.ReLU()(x)
        x=tf.reshape(x,[-1,middle_output,1])
        x_glu_no_sig = tf.keras.layers.Conv1D(1,3,padding='same')(x)
        x_glu_no_sig=tf.reshape(x_glu_no_sig,[-1,middle_output])

        x_glu = tf.keras.layers.Conv1D(1,3,padding='same',activation='sigmoid')(x)
        x_glu=tf.reshape(x_glu,[-1,middle_output])

        x_glu_2_step_1=tf.math.multiply(x_glu_no_sig,x_glu)
        x_glu_2_step_1=tf.reshape(x_glu_2_step_1,[-1, middle_output])
        x=tf.keras.layers.Dense(middle_output)(x_glu_2_step_1)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.ReLU()(x)
        x=tf.reshape(x,[-1,middle_output,1])
        x_glu_no_sig = tf.keras.layers.Conv1D(1,3,padding='same')(x)
        x_glu_no_sig=tf.reshape(x_glu_no_sig,[-1,middle_output])

        x_glu = tf.keras.layers.Conv1D(1,3,padding='same',activation='sigmoid')(x)
        x_glu=tf.reshape(x_glu,[-1,middle_output])

        x_glu_2_step_2=tf.math.multiply(x_glu_no_sig,x_glu)

        xx=tf.math.add(x_glu_2_step_1,x_glu_2_step_2)
        out=tf.math.multiply(xx,tf.math.sqrt(0.5))
        #return out
        return tf.keras.models.Model(inputs=inputs,outputs=out)
    def no_shared_decision(self,inputs,no_shared_num):
        x=tf.keras.layers.Dense(no_shared_num)(inputs)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.ReLU()(x)
        x=tf.reshape(x,[-1,no_shared_num,1])
        x_glu_no_sig = tf.keras.layers.Conv1D(1,3,padding='same')(x)
        x_glu_no_sig =tf.reshape(x_glu_no_sig,[-1,no_shared_num])

        x_glu = tf.keras.layers.Conv1D(1,3,padding='same',activation='sigmoid')(x)
        x_glu =tf.reshape(x_glu,[-1,no_shared_num])

        x_glu_2_step_1=tf.math.multiply(x_glu_no_sig,x_glu)        
        x_1=tf.math.add(inputs,x_glu_2_step_1)
        x_1=tf.math.multiply(x_1,tf.math.sqrt(0.5))

        x=tf.keras.layers.Dense(no_shared_num)(x_1)
        x=tf.keras.layers.BatchNormalization()(x)
        x=tf.keras.layers.ReLU()(x)
        x=tf.reshape(x,[-1,no_shared_num,1])


        x_glu_no_sig = tf.keras.layers.Conv1D(1,3,padding='same')(x)
        x_glu_no_sig =tf.reshape(x_glu_no_sig,[-1,no_shared_num])

        x_glu = tf.keras.layers.Conv1D(1,3,padding='same',activation='sigmoid')(x)
        x_glu =tf.reshape(x_glu,[-1,no_shared_num])

        x_glu_2_step_2=tf.math.multiply(x_glu_no_sig,x_glu)
        xx=tf.math.add(x_glu_2_step_1,x_glu_2_step_2)
        out=tf.math.multiply(xx,tf.math.sqrt(0.5))
        return out


    def Attentive_transformer(self,inputs,feature_num,prior_scale__):
        x=tf.keras.layers.Dense(feature_num)(inputs)
        x=tf.keras.layers.BatchNormalization()(x)

        prior_scale=tf.reshape(prior_scale__,[-1,x.shape[1]])


        x=tf.keras.layers.Multiply()([x,prior_scale])
        #x=tf.multiply(x,prior_scale)
        x=tfa.activations.sparsemax(x)
        return x,prior_scale

    def feature_transformer(self,inputs,sh_dt):
        x=sh_dt(inputs)
        x=self.no_shared_decision(x,self.ds_node_num)
        return x
    def make_model(self,input_sh):
        x_input=tf.keras.layers.Input(shape=(input_sh))
        #self.prior_scale=tf.cast(tf.math.equal(x_input, x_input), tf.float32)
        x_in=tf.keras.layers.BatchNormalization()(x_input)
        shared_dt=self.shared_decision(x_in.shape[1],self.ds_node_num)
        x=self.feature_transformer(x_in,shared_dt)
        #prior_scale=tf.constant(np.ones((1, x_input.shape[1])))
        for ds in range(self.step_num):
            mask, self.prior_scale=self.Attentive_transformer(x,x_input.shape[1], self.prior_scale)
            self.prior_scale=tf.keras.layers.Multiply()([(self.ramda-mask),self.prior_scale])
            #self.prior_scale=tf.multiply(self.prior_scale,1)
            x=tf.multiply(x_in,mask)
            x=self.feature_transformer(x,shared_dt)
            x_for_de=tf.identity(x)
            if ds==0:
                for_last_de=tf.keras.layers.ReLU()(x_for_de)
            else:
                for_last_de=tf.add(tf.keras.layers.ReLU()(x_for_de),for_last_de)
        out=tf.keras.layers.Dense(self.last_mlp_n)(for_last_de)
        out=tf.keras.layers.BatchNormalization()(out)
        out=tf.keras.layers.ReLU()(out)

        out=tf.keras.layers.Dense(self.class_num,activation='softmax')(out)
        model=tf.keras.models.Model(inputs=x_input,outputs=out)
        return model 

            


batch_s=32






tabnet=tabnet_classifier(train_x.shape[1],64,16,2,len(list(set(train_lbls))),ramda=1.3)
tabnet_m=tabnet.make_model(train_x.shape[1])
opt=optimizers.Adam(lr=0.001)
tabnet_m.compile(optimizer=opt,loss='sparse_categorical_crossentropy')
for i in range(20):
    tabnet_m.fit(train_x,train_lbls,epochs=1,batch_size=batch_s)
    pred_x=tabnet_m.predict(train_x,verbose=0)
    preds=[]
    preds_p=[]
    idx_test=0
    for pred in pred_x:
        preds.append(pred.argmax())
        preds_p.append(pred[1])
        idx_test+=1

    result = Counter(preds)
    print(result)
    result = Counter(list(train_lbls))
    print('real train')
    print(result)
    for key in result:
        print(key, result[key])
    train_x.index=list(range(train_x.shape[0]))
    bacc=balanced_accuracy_score(list(train_lbls),preds)
    acc=accuracy_score(list(train_lbls),preds)


    print('\ntrain')
    print('bacc: ',bacc)
    print('acc: ',acc)
    print()
    pred_x=tabnet_m.predict(test_x,verbose=0)

    preds=[]
    preds_p=[]
    idx_test=0
    for pred in pred_x:
        preds.append(pred.argmax())
        preds_p.append(pred[1])
        idx_test+=1

    result = Counter(preds)
    print(result)
    result = Counter(list(test_lbls))
    print('real test')
    print(result)
    for key in result:
        print(key, result[key])
    test_x.index=list(range(test_x.shape[0]))
    bacc=balanced_accuracy_score(list(test_lbls),preds)
    acc=accuracy_score(list(test_lbls),preds)


    print('\ntest')
    print('bacc: ',bacc)
    print('acc: ',acc)

    





