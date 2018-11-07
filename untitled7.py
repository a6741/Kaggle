# coding: utf-8



#如果要在Python 2.7的代码中直接使用Python 3.x的语法，可以通过__future__引入对应的模块
from __future__ import absolute_import
from __future__ import division
#import argparse

#是只引入tensorflow.examples.tutorials.mnist包里的input_data类
from tensorflow.examples.tutorials.mnist import input_data

#给tensorflow包一个别称tf
import tensorflow as tf



def sigmaprime(x):
    return tf.multiply(tf.sigmoid(x),tf.subtract(tf.constant(1.0),tf.sigmoid(x)))




def variable_summaries(var, name):
    with tf.name_scope('summary_'+name):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)  #输出平均值
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev) #输出标准差
        tf.summary.scalar('max',tf.reduce_max(var)) #输出最大值
        tf.summary.scalar('min',tf.reduce_min(var)) #输出最小值
        tf.summary.histogram('histogram',var) #输出柱状图




#定义神经网络的结构相关参数
ETA = 0.01   #学习率
EPOCHS = 2000  #训练次数
BATCH_SIZE = 1000    #批量数
TEST_EPOCHS = 10  #测试间隔

INPUT_NODE = 12 #输入节点数
OUTPUT_NODE = 2    #输出的节点数

#通过改变下面这个参数来改变中间神经元的个数
HIDDENLAYER_NODE = 30    #隐藏层的节点数
LOG_PATH='log/mnist_bp_2Layer/'  #图输出的目录
import titanic



#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
thex,thep,they=titanic.gettrain()


# 输入的图片
x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
# 输入图片的标签
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

#第一层
W_1 = tf.Variable(tf.zeros([INPUT_NODE, HIDDENLAYER_NODE]))
b_1 = tf.Variable(tf.zeros(HIDDENLAYER_NODE))
z_1 = tf.matmul(x, W_1) + b_1
y_1 = tf.sigmoid(z_1)
    

#第二层
W_2 = tf.Variable(tf.zeros([HIDDENLAYER_NODE, OUTPUT_NODE]))
b_2 = tf.Variable(tf.zeros(OUTPUT_NODE))
z_2 = tf.matmul(y_1, W_2) + b_2
y_2 = tf.sigmoid(z_2)




quadratic_cost = tf.subtract(y_2,y_)





d_z2 = tf.multiply(quadratic_cost, sigmaprime(z_2))
d_b2 = d_z2
d_w2 = tf.matmul(tf.transpose(y_1),d_z2)


# ## 第一层的修改值



d_z1 = tf.multiply(tf.matmul(d_z2,tf.transpose(W_2)), sigmaprime(z_1))
d_b1 = d_z1
d_w1 = tf.matmul(tf.transpose(x),d_z1)
    
step = [
    tf.assign(W_1,
            tf.subtract(W_1, tf.multiply(ETA, d_w1)))
  , tf.assign(b_1,
            tf.subtract(b_1, tf.multiply(ETA,
                               tf.reduce_mean(d_b1, axis=[0]))))
  , tf.assign(W_2,
            tf.subtract(W_2, tf.multiply(ETA, d_w2)))
  , tf.assign(b_2,
            tf.subtract(b_2, tf.multiply(ETA,
                               tf.reduce_mean(d_b2, axis=[0]))))
]




accuracy_mat = tf.equal(tf.argmax(y_2,1),tf.argmax(y_,1))
accuracy_result = tf.reduce_mean(tf.cast(accuracy_mat,tf.float32))


#第一层
with tf.name_scope('layer1'):
    variable_summaries(W_1,'W_1')
    variable_summaries(b_1,'b_1')
    
#第二层
with tf.name_scope('layer2'):
    variable_summaries(W_2,'W_2')
    variable_summaries(b_2,'b_2')

with tf.name_scope('Accuracy'):
    tf.summary.scalar('accuracy_rate',accuracy_result)
    
#把所有的summary合到一张图上
merged = tf.summary.merge_all()

#设置训练和测试的Writer
train_writer = tf.summary.FileWriter(LOG_PATH+'/train_'+str(HIDDENLAYER_NODE)+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(ETA),graph=tf.get_default_graph())
test_writer = tf.summary.FileWriter(LOG_PATH+'/test_'+str(HIDDENLAYER_NODE)+'_'+str(BATCH_SIZE)+'_'+str(EPOCHS)+'_'+str(ETA),graph=tf.get_default_graph())


with tf.Session() as sess:
    # 初始化之前定义好的全部变量
    tf.global_variables_initializer().run()
  
    #对模型训练EPOCHS次
    #随机选取BATCH_SIZE个图像数据进行训练
    for i in range(EPOCHS):
        #batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        summary,_=sess.run([merged,step], feed_dict={x: thex, y_: they})
        # 把Summary加入到训练数据的Writer中
        train_writer.add_summary(summary,i)
        if i % TEST_EPOCHS == 0 :
            summary = sess.run(merged,feed_dict={x: thex, y_: they})
            test_writer.add_summary(summary,i)
    y_2=sess.run(y_2,feed_dict={x: thep, y_: they})

import numpy as np
import pandas as pd
y_3=np.argmax(y_2,axis=1)
tfi=pd.read_csv('/home/ljk/下载/kaggle/all/test.csv',encoding='utf8',index_col='PassengerId')
tfi.loc[:,'Survived']=y_3
tu=tfi.loc[:,'Survived']
tu.to_csv('pr.csv',header=True)
train_writer.close()
test_writer.close()

