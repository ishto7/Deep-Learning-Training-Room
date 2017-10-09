from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file= 'D:\\Coding\\notMNIST.pickle'
with open(pickle_file,'rb') as f:
    save=pickle.load(f)
    train_dataset=save['train_dataset']
    train_labels=save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 32
patch_size = 3
depth = 32
num_hidden = 128
num_hidden2 = 22

graph=tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size , image_size,num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    size3 = ((image_size-patch_size+1)//2-patch_size+1)//2
    layer3_weights = tf.Variable(tf.truncated_normal(
        [size3*size3 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    layer5_weights = tf.Variable(tf.truncated_normal(
        [num_hidden2, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    def model(data,keep_p1:tf.float32=1,keep_p2:tf.float32=1):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.max_pool(tf.nn.relu(conv + layer1_biases),[1,2,2,1],[1,2,2,1],padding="VALID")
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.max_pool(tf.nn.relu(conv + layer2_biases),[1,2,2,1],[1,2,2,1],padding="VALID")
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        drop = tf.nn.dropout(reshape,keep_prob=keep_p1)
        hidden = tf.nn.relu(tf.matmul(drop, layer3_weights) + layer3_biases)
        drop = tf.nn.dropout(hidden, keep_prob=keep_p2)
        return tf.nn.softmax(tf.matmul(drop, layer4_weights) + layer4_biases)

    logits = model(tf_train_dataset,0.25,0.5)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    train_prediction = logits
    valid_prediction = model(tf_valid_dataset)
    test_prediction = model(tf_test_dataset)


num_steps=10001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Hey Run")
    for step in range(num_steps):
        offset= (step* batch_size)%(train_labels.shape[0]-batch_size)
        batch_data=train_dataset[offset:offset+batch_size,:,:,:]
        batch_labels= train_labels[offset:offset+batch_size,:]
        feed_dict={tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        _,l,predictions=session.run([optimizer,loss,train_prediction], feed_dict=feed_dict)
        if (step%1000)==0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))