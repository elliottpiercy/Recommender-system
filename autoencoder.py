import numpy as np
import tensorflow as tf
import os

class autoencoder:

    def __init__(self,batch_size,num_hid1,num_hid2,activation):
        self.batch_size = batch_size # 20
        self.num_hid1 = num_hid1 #256
        self.num_hid2 = num_hid2 #128
        self.num_hid3 = num_hid1
        self.activation = activation


    def act(self):
        if self.activation == 'relu':
            return tf.nn.relu
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid
        elif self.activation == 'tanh':
            return tf.nn.tanh


    def network_weights(self):
        initializer=tf.variance_scaling_initializer()
        w1=tf.Variable(initializer([self.batch_size,self.num_hid1]),dtype=tf.float32)
        w2=tf.Variable(initializer([self.num_hid1,self.num_hid2]),dtype=tf.float32)
        w3=tf.Variable(initializer([self.num_hid2,self.num_hid3]),dtype=tf.float32)
        w4=tf.Variable(initializer([self.num_hid3,self.batch_size]),dtype=tf.float32)
        return w1,w2,w3,w4


    def network_biases(self):
        b1=tf.Variable(tf.zeros(self.num_hid1))
        b2=tf.Variable(tf.zeros(self.num_hid2))
        b3=tf.Variable(tf.zeros(self.num_hid3))
        b4=tf.Variable(tf.zeros(self.batch_size))
        return b1,b2,b3,b4


    def mse(self,x,y):
        return np.mean((x-y)**2)


    def predict(self,X_test):
        err = np.array([])
        num_batches= int(len(X_test)/self.batch_size)
        for batch in range(num_batches):
            X_batch = X_test[batch:batch+self.batch_size].T
            test_results = output_layer.eval(feed_dict={X:batch})

    def build_graph(self):
        tf.reset_default_graph()

        actf = self.act()
        X = tf.placeholder(tf.float32,shape=[None,self.batch_size])

        #weights for the network
        w1,w2,w3,w4 = self.network_weights()
        weights = w1,w2,w3,w4

        # biases for for the network
        b1,b2,b3,b4 = self.network_biases()
        bias = b1,b2,b3,b4

        #layer activation functions
        hid_layer1 = actf(tf.matmul(X,w1)+b1)
        hid_layer2 = actf(tf.matmul(hid_layer1,w2)+b2)
        hid_layer3 = actf(tf.matmul(hid_layer2,w3)+b3)
        output_layer=actf(tf.matmul(hid_layer3,w4)+b4)

        act_fns = hid_layer1, hid_layer2, hid_layer3,output_layer
        loss=tf.reduce_mean(tf.square(output_layer-X))

        return X, weights, bias, act_fns,loss

    def fit(self,save_path,save_name,lr,epochs,X_train):



        X, weights, bias, act_fns,loss = self.build_graph()
        w1,w2,w3,w4 = weights
        b1,b2,b3,b4 = bias
        hid_layer1, hid_layer2, hid_layer3,output_layer = act_fns
        #loss fn = mean squared error

        optimizer=tf.train.AdamOptimizer(lr)
        train=optimizer.minimize(loss)

        saver = tf.train.Saver()
        init=tf.global_variables_initializer()

        # training phase
        num_batches= int(len(X_train)/self.batch_size)
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(epochs):

                for batch in range(num_batches):
                    X_batch = X_train[batch*self.batch_size:(batch+1)*self.batch_size].T
                    sess.run(train,feed_dict={X:X_batch})

                train_loss=loss.eval(feed_dict={X:X_batch})

                print("epoch {} loss {}".format(epoch+1,train_loss))

            save_path = saver.save(sess, os.path.join(save_path, save_name))



    def predict(self,save_path,save_name,X_test):

        X, weights, bias, act_fns,loss = self.build_graph()
        w1,w2,w3,w4 = weights
        b1,b2,b3,b4 = bias
        hid_layer1, hid_layer2, hid_layer3,output_layer = act_fns

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            saver = saver.restore(sess, os.path.join(save_path,save_name))

            err = np.array([])
            predictions = np.array([])
            num_batches= int(len(X_test)/self.batch_size)

            for batch in range(num_batches):
                X_batch = X_test[batch*self.batch_size:(batch+1)*self.batch_size].T

                test_results = output_layer.eval(feed_dict={X:X_batch})

                predictions = np.append(predictions,test_results)
                err = np.append(err,self.mse(X_batch,test_results))


        return np.mean(err),np.reshape(predictions,X_test.shape)
