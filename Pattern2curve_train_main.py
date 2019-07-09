# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:28:33 2019

@author: Hongqinag Zhou 
#emails: photonzhou@163.com
"""
# this is a simple structure from pattern to transmission curve. And wavelength is range from 700nm to 1600nm. The train
# data and label from RCWA based on Nano-Optics. It`s details are as fellow:...
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data #导入数据的模块
from scipy.io import loadmat
train_data_inputsPattern_path='./train_10_10_data_curve/train_data_inputsPattern.mat' #read train pattern path
train_data_curve_x_label_path='./train_10_10_data_curve/train_data_curve_x_label.mat' #read train x label path

data_patt=loadmat(train_data_inputsPattern_path)
train_data_inputsPattern_all=data_patt['train_data_inputsPattern'] 

data_curve=loadmat(train_data_curve_x_label_path)
train_data_curve_x_label_all=data_curve['train_data_curve_x_label']

train_data_inputsPattern_all = np.asarray(train_data_inputsPattern_all,'float32')
train_data_curve_x_label_all = np.asarray(train_data_curve_x_label_all,'float32')

train_data_inputsPattern_all = np.reshape(train_data_inputsPattern_all, (-1,10000))
train_data_inputsPattern_all = np.transpose(train_data_inputsPattern_all)
train_data_curve_x_label_all = np.transpose(train_data_curve_x_label_all)

train_data1_inputsPattern_path='./train_10_10_data_curve/train_data1_inputsPattern.mat' #read train pattern path
train_data1_curve_x_label_path='./train_10_10_data_curve/train_data1_curve_x_label.mat' #read train x label path

data1_patt=loadmat(train_data1_inputsPattern_path)
train_data1_inputsPattern_all=data1_patt['train_data1_inputsPattern'] 

data1_curve=loadmat(train_data1_curve_x_label_path)
train_data1_curve_x_label_all=data1_curve['train_data1_curve_x_label']

train_data1_inputsPattern_all = np.asarray(train_data1_inputsPattern_all,'float32')
train_data1_curve_x_label_all = np.asarray(train_data1_curve_x_label_all,'float32')

train_data1_inputsPattern_all = np.reshape(train_data1_inputsPattern_all, (-1,10000))
train_data1_inputsPattern_all = np.transpose(train_data1_inputsPattern_all)
train_data1_curve_x_label_all = np.transpose(train_data1_curve_x_label_all)

train_X=train_data_inputsPattern_all[0:10000,:] # training data pattern 1-8000 pics
train_Y=train_data_curve_x_label_all[0:10000,:] # training label 1-8000  pics

train_X = np.vstack((train_X,train_data1_inputsPattern_all))
train_Y = np.vstack((train_Y,train_data1_curve_x_label_all))

test_X=train_X[19000:20000,:] #testing label 2000 pics
test_Y=train_Y[19000:20000,:]# testing label 2000 pics



train_X = train_X[0:9000]
train_Y = train_Y[0:9000]


# show data patterns and labels
wl = list(range(750,1600,20))
#
##plt.figure()
##plt.imshow(train_data_inputsPattern_all[:,:,108])
##plt.figure()
##plt.plot(wl,train_data_curve_x_label_all[:,108],label='t label')
##plt.xlabel('wavelength')
##plt.ylabel('transmission')
##plt.legend(loc='upper right')
#
##创建一个简单的网络
batch_size=20  #表示每一批训练100组数据，因为训练集共有数据10000组，故而训练一个周期需要经过100次迭代
test_size=10  #作为验证数据，验证集有10000组数据，但这里只验证256组，因为数据太多，运算太慢
img_size=10     #nanostructure的大小
num_class=43    #曲线取点数
train_epochs = 1001      #训练轮数
lr = 0.0001 #learning rate
keep_prob = 1   #dropout ratio
n_samples = len(train_X)
# creat palcehoder
X = tf.placeholder(tf.float32,[None,100])
Y = tf.placeholder(tf.float32,[None,43])


n_batch = n_samples // batch_size # 批次大小
keep_prob = tf.placeholder(tf.float32)


def get_random_batchdata(n_samples, batch_size):
    start_index = np.random.randint(0, n_samples - batch_size)
    return (start_index, start_index + batch_size)

#define weigth and bias
W1 = tf.Variable(tf.random_normal([100,200],stddev=0.001)/100)
B1 = tf.Variable(tf.zeros([200])+0.01)
L1 = tf.nn.relu(tf.matmul(X,W1)+B1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.random_normal([200,300],stddev=0.001)/100)
B2 = tf.Variable(tf.zeros([300])+0.01)
L2 = tf.nn.relu(tf.matmul(L1_drop,W2)+B2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.random_normal([300,500],stddev=0.001)/100)
B3 = tf.Variable(tf.zeros([500])+0.01)
L3 = tf.nn.relu(tf.matmul(L2_drop,W3)+B3)
L3_drop = tf.nn.dropout(L3,keep_prob)
#
W4 = tf.Variable(tf.random_normal([500,500],stddev=0.001)/100)
B4 = tf.Variable(tf.zeros([500])+0.01)
L4 = tf.nn.relu(tf.matmul(L3_drop,W4)+B4)
L4_drop = tf.nn.dropout(L4,keep_prob)
#
W5 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1)/100)
B5 = tf.Variable(tf.zeros([300])+0.01)
L5 = tf.nn.relu(tf.matmul(L4_drop,W5)+B5)
L5_drop = tf.nn.dropout(L5,keep_prob)
#
#W6 = tf.Variable(tf.truncated_normal([200,200],stddev=0.1)/100)
#B6 = tf.Variable(tf.zeros([200])+0.1) 
#L6 = tf.nn.relu(tf.matmul(L5_drop,W6)+B6)
#L6_drop = tf.nn.dropout(L6,keep_prob)
#
#W7 = tf.Variable(tf.truncated_normal([200,200],stddev=0.1)/100)
#B7 = tf.Variable(tf.zeros([200])+0.1)
#L7 = tf.nn.relu(tf.matmul(L6_drop,W7)+B7)
#L7_drop = tf.nn.dropout(L7,keep_prob)
#
#W8 = tf.Variable(tf.truncated_normal([200,200],stddev=0.1)/100)
#B8 = tf.Variable(tf.zeros([200])+0.1)
#L8 = tf.nn.relu(tf.matmul(L7_drop,W8)+B8)
#L8_drop = tf.nn.dropout(L8,keep_prob)
#
#W9 = tf.Variable(tf.truncated_normal([200,200],stddev=0.1)/100)
#B9 = tf.Variable(tf.zeros([200])+0.1)
#L9 = tf.nn.relu(tf.matmul(L8_drop,W9)+B9)
#L9_drop = tf.nn.dropout(L9,keep_prob)

W10 = tf.Variable(tf.random_normal([300,43],stddev=0.001)/100)
B10 = tf.Variable(tf.zeros([43])+0.01)
# prediction outpout 
prediction = tf.matmul(L5_drop,W10)+B10


#激活输出
#prediction = tf.nn.softmax(prediction)
#二次代价函数
Loss = tf.reduce_mean(tf.square(Y-prediction))
#Loss = tf.reduce_mean(abs(Y-prediction))
#Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction))

#使用梯度下降函数

#train_step = tf.train.GradientDescentOptimizer(lr).minimize(Loss)
#train_step = tf.train.AdagradOptimizer(lr).minimize(Loss)
train_step = tf.train.AdamOptimizer(lr).minimize(Loss)
#
#初始化变量
init = tf.global_variables_initializer()

# 直接输出预测曲线
#求准确率
#accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

#创建会话
with tf.Session()  as sess:
    sess.run(init)
    a_loss=[]
    a_test_acc=[]
    for epoch in range(train_epochs):
        
        for batch in range(n_batch):
            start_index, end_index = get_random_batchdata(n_samples, batch_size)
    
            batch_xs = train_X[start_index: end_index]
            batch_ys = train_Y[start_index: end_index]
            _, train_loss = sess.run([train_step, Loss],feed_dict={X:batch_xs,Y:batch_ys,keep_prob:0.8})
    

#            train_acc = sess.run(accuracy,feed_dict={X:train_X,Y:train_Y,keep_prob:1.0})
        train_loss=train_loss*1000  #此处没有任何意义，不影响程序运行
        a_loss.append(train_loss)#保存损失函数数组
        print(" Epoch "+ str(epoch)+ ", Training loss " + str (train_loss))
#    test_image = test_X[111:112]
#    test_label = test_Y[111:112]
#    test_result = sess.run(prediction,feed_dict={X:test_image,keep_prob:0.7})   
#    _, test_loss = sess.run([train_step, Loss],feed_dict={X:test_image,Y:test_label,keep_prob:0.7})
#    print ('Testing loss is: ' + str(test_loss*1000))    
#    plt.figure()
#    test_result = np.transpose(test_result)
#    plt.plot(wl,test_Y[111],label='t label',linewidth='2',color='b')
#    plt.plot(wl,test_result,label='predicton',linewidth='2',color='r')   

    print ('training finished')
    plt.figure()
    plt.show()#绘图损失函数
    plt.plot(a_loss)
    plt.xlabel('Epoch')
    plt.ylabel('loss function')
    print('===================下面是保存模型================================')
         #保存模型
    saver=tf.train.Saver()

    saver.save(sess,'model/curvePRE_model2.ckpt')
    path=saver.save(sess,'model/curvePRE_model2.ckpt')
    print('模型欧保存到: %s' %(path),end='\n')
    print('Model saving finished！')      
#
#
#
#
test_image = test_X[118:119]
test_label = test_Y[118:119]
#X = tf.placeholder(tf.float32,[None,100])
#Y = tf.placeholder(tf.float32,[None,43])

with tf.Session() as sess:   #
    t_loss = []
    new_saver=tf.train.import_meta_graph('model/curvePRE_model2.ckpt.meta') #第二步：导入模型的图结构
    new_saver.restore(sess,'model/curvePRE_model2.ckpt')  #第三步：将这个会话绑定到导入的图中
    print('=======================模型加载完成=============================')
#    sess.run(tf.global_variables_initializer())  
    test_result = sess.run(prediction,feed_dict={X:test_image,keep_prob:1})
    _, test_loss = sess.run([train_step, Loss],feed_dict={X:test_image,Y:test_label,keep_prob:0.8})
    
    print ('Testing loss is: ' + str(test_loss*1000))

   

plt.figure()
plt.imshow(np.reshape(test_image, [10, 10]), cmap='gray')
plt.title('test image')

plt.figure()
test_result = np.transpose(test_result)
test_label = np.transpose(test_label)
plt.plot(wl,test_label,label='t label',linewidth='2',color='b')
plt.plot(wl,test_result,label='predicton',linewidth='2',color='r')

plt.xlabel('Wavelength')
plt.ylabel('Transmission')
plt.legend(loc='upper right')
#    
#    X = tf.placeholder(tf.float32,[None,100])
#    Y = tf.placeholder(tf.float32,[None,43])
#    
#    start_index, end_index = get_random_batchdata(n_samples, test_size)
#    
#    batch_xs = test_X[start_index: end_index]
#    batch_ys = test_Y[start_index: end_index]
#    test_loss = sess.run(Loss, feed_dict={X:batch_xs,Y:batch_ys,keep_prob:1.0})
#    print('测试集上面的loss为 ：%f'%(test_loss),end='\n')
























#卷积神经网络结构，后备用
##第一个卷积层
   
#w1=tf.Variable(tf.random_normal(shape=[10,10,1,32],stddev=0.01))
#conv1=tf.nn.conv2d(X,w1,strides=[1,1,1,1],padding="SAME")
#conv_y1=tf.nn.relu(conv1)

##第一个池化层
#
#pool_y2=tf.nn.max_pool(conv_y1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#pool_y2=tf.nn.dropout(pool_y2,p_keep)
#
##第二个卷积层
#
#w2=tf.Variable(tf.random_normal(shape=[3,3,32,64],stddev=0.01))
#conv2=tf.nn.conv2d(pool_y2,w2,strides=[1,1,1,1],padding="SAME")
#conv_y3=tf.nn.relu(conv2)
#
##第二个池化层
#
#pool_y4=tf.nn.max_pool(conv_y3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#pool_y4=tf.nn.dropout(pool_y4,p_keep)
#
##第三个卷积层
#
#w3=tf.Variable(tf.random_normal(shape=[3,3,64,128],stddev=0.01))
#conv3=tf.nn.conv2d(pool_y4,w3,strides=[1,1,1,1],padding="SAME")
#conv_y5=tf.nn.relu(conv3)
#
##第三个池化层
#
#pool_y6=tf.nn.max_pool(conv_y5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
##全连接层 
#
#w4=tf.Variable(tf.random_normal(shape=[128*4*4,625],stddev=0.01))
#FC_layer=tf.reshape(pool_y6,[-1,w4.get_shape().as_list()[0]])
#FC_layer=tf.nn.dropout(FC_layer,p_keep)
#FC_y7=tf.matmul(FC_layer,w4)
#FC_y7=tf.nn.relu(FC_y7)
#FC_y7=tf.nn.dropout(FC_y7,p_keep)
#
##输出层，model_Y则为神经网络的预测输出
#
#w5=tf.Variable(tf.random_normal(shape=[625,num_class]))
#model_Y=tf.matmul(FC_y7,w5,name='output')

##损失函数
#Y_=tf.nn.softmax_cross_entropy_with_logits(logits=model_Y,labels=Y)
#cost=tf.reduce_mean(Y_)
#
##准确率
#correct_prediction=tf.equal(tf.argmax(model_Y,axis=1),tf.argmax(Y,axis=1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
##优化方式
##optimizer=tf.train.RMSPropOptimizer(learn_rate,drop_out).minimize(cost)     ##RMSPropOptimizer优化器
#optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)   ##adam优化器

























