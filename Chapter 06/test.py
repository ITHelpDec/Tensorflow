import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

batch_size = 64
embedding_dimension = 5
negative_samples = 8
LOG_DIR = "tensorflow/logs/word2vec_intro"

digit_to_word_map = {1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}
sentences = []

# Create two kinds of sentences - sequences of odd and even digits
for i in range(10000):
    rand_odd_ints = np.random.choice(range(1,10,2),3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2,10,2),3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

# Map words to indices
word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        # lower() - make all words lowercase
        # split() - turns string into array / 1x3 matrix
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
index2word_map = {index:word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map) # 9

# Generate skip-gram pairs
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent)-1): # range is 1-2
        word_context_pair = [[word2index_map[tokenized_sent[i-1]], word2index_map[tokenized_sent[i+1]]], word2index_map[tokenized_sent[i]]]
        # output example [[3, 7], 3]
        # [3, 7]
        # [3, 0]

        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][0]])
        # should add [3, 3] to array but came up as [1, 0]
        skip_gram_pairs.append([word_context_pair[1], word_context_pair[0][1]])
        # should add [3, 7] to array, but comes up as [1, 2]

def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs))) #40,000
    np.random.shuffle(instance_indices) # "None"
    batch = instance_indices[:batch_size] # do first 64 instances
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y

# skip_gram_pairs[0:10]
# [[1, 0],
#  [1, 2],
#  [3, 3],
#  [3, 3]
#  [1, 2],
#  [1, 4],
#  [6, 5],
#  [6, 5],
#  [4, 1],
#  [4, 7]]

# Batch example
# x_batch, y_batch = get_skipgram_batch(8)
# x_batch
# y_batch
# [index2word_map[word] for word in x_batch]
# [index2word_map[word[0]] for word in y_batch]

with tf.name_scope("data"):
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1.0, 1.0), name = 'embedding')
        # creates random number between -1 and 1 of the same shape as the vocabulary_size i.e. scalar
    # This is essentially a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Create variables for the NCE loss
with tf.name_scope("loss"):
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dimension], stddev=1.0 / math.sqrt(embedding_dimension)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases = nce_biases,
                                         inputs = embed,
                                         labels = train_labels,
                                         num_sampled = negative_samples,
                                         num_classes = vocabulary_size))

# Learning rate decay
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# Merge all summary ops
#merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    total_summary = 0.0
    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k,v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    # Link embedding to its metadata file
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)

    # tf.global_variables_initializer().run()

    for step in range(5000):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        # summary,_ = sess.run([merged, train_step], feed_dict={train_inputs:x_batch, train_labels:y_batch})
        summary,_ = sess.run([loss, train_step], feed_dict={train_inputs:x_batch, train_labels:y_batch})
        # train_writer.add_summary(summary, step)
        total_summary += summary

        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss, feed_dict={train_inputs:x_batch, train_labels:y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))

    # Normalise embeddings before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

ref_word = normalized_embeddings_matrix[word2index_map["one"]]

cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print(index2word_map[f])
    print(cosine_dists[f])
