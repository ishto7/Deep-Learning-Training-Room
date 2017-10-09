import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", False,True)

input_neurons = tf.placeholder(tf.float32,[None, 784])

hidden_layer1_n=300

W1=tf.Variable(tf.truncated_normal([784, hidden_layer1_n]))
b1=tf.Variable(tf.zeros([hidden_layer1_n]))
hn1=tf.nn.relu(tf.matmul(input_neurons,W1) + b1)

W2=tf.Variable(tf.truncated_normal([hidden_layer1_n, 10]))
b2=tf.Variable(tf.zeros([10]))
output=tf.nn.softmax(tf.matmul(hn1,W2) + b2)


target=tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)) +0.01*(
    tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2)+ tf.nn.l2_loss(b1))
trainstep=tf.train.AdagradOptimizer(0.5).minimize(cross_entropy)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()


for step in range(2000):
    batch_x, batch_y= mnist.train.next_batch(100)
    sess.run(trainstep,feed_dict={input_neurons:batch_x,target:batch_y})
    if (step % 500 == 0):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Validation accuracy at step %d: %f" % (step,sess.run(accuracy, feed_dict={input_neurons: mnist.test.images, target: mnist.test.labels})))

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={input_neurons: mnist.test.images, target: mnist.test.labels}))
