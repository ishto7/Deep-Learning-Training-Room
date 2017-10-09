from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import arabic_reshaper
from bidi.algorithm import get_display
def make_farsi_text(x):
    reshaped_text = arabic_reshaper.reshape(x)
    farsi_text = get_display(reshaped_text)
    return farsi_text

filename = 'D:\\Coding\\tutorial\\mrshabanali3.txt'
fin2=open(filename,'r',encoding='utf-8').read()
words=fin2.split()
print('Data size %d' % len(words))

vocabulary_size = 10000

def build_dataset(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            data.append(dictionary[word])
        else:
            data.append(0)
            unk_count=unk_count+1
    count[0][1]=unk_count
    reverese_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverese_dictionary

data,count,dictionary,reverese_dictionary=build_dataset(words)
print(data[:10])
del words
data_index=0

def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size % num_skips ==0
    assert num_skips <=2*skip_window
    batch= np.ndarray(shape=(batch_size),dtype=np.int32)
    labels= np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span= 2*skip_window+1
    buffer=collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range (batch_size//num_skips):
        target= skip_window
        targets_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[j+i*num_skips]=buffer[skip_window]
            labels[j+i*num_skips,0]=buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)

    return batch, labels


print('data:', [reverese_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverese_dictionary[bi] for bi in batch])
    print('    labels:', [reverese_dictionary[li] for li in labels.reshape(8)])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(),tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embeddings=tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
    softmax_weights =tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    embed=tf.nn.embedding_lookup(embeddings,train_dataset)
    loss=tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights,biases=softmax_biases,inputs=embed,labels=train_labels,num_sampled=num_sampled,num_classes=vocabulary_size))

    optimizer =tf.train.AdagradOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps= 100001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Hey Buddy')
    average_loss=0
    for step in range(num_steps):
        batch_data, batch_labels= generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_dataset:batch_data,train_labels:batch_labels}
        _,l=session.run([optimizer,loss],feed_dict=feed_dict)
        average_loss+=l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        if step%10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_words= reverese_dictionary[valid_examples[i]]
                top_k=8
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log='nearest to %s'% valid_words
                for k in range(top_k):
                    close_word = reverese_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()


num_points = 500
tsne= TSNE(perplexity=30,init='pca',n_iter=5000,method='exact')
two_d_embedding=tsne.fit_transform(final_embeddings[:num_points,:])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(make_farsi_text(label), xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverese_dictionary[i] for i in range( num_points)]
plot(two_d_embedding, words)
