import tensorflow as tf
import numpy as np
import input_data
import pickle
# open_file_name = 'weights_log/weights10.pkl'
# open_file_name = 'weights_log/weights_quan'+'.pkl'
# open_file_name = 'weights10'+'.pkl'
open_file_name = 'weights_log/pcov92pfc92'+'.pkl'
# open_file_name = 'weights_log/weights0.pkl'
Test = True;
# MASK_GEN = True
MASK_GEN = False
sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)

def mask_gen():
    #generate mask based on weights
    with open(open_file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    # print(wc1)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': wc1,
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': wc2,
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': wd1,
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': out
    }

    biases = {
        'cov1': bc1,
        'cov2': bc2,
        'fc1': bd1,
        'fc2': bout
    }
    keys = ['cov1', 'cov2', 'fc1', 'fc2']
    masks = {}
    for key in keys:
        masks[key] = weights[key] != 0
    with open('mask.pkl', 'wb') as f:
        pickle.dump(masks,f)





def initialize_variables():
    with open(open_file_name,'rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': tf.Variable(wc1),
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': tf.Variable(wc2),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': tf.Variable(wd1),
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': tf.Variable(out)
    }

    biases = {
        'cov1': tf.Variable(bc1),
        'cov2': tf.Variable(bc2),
        'fc1': tf.Variable(bd1),
        'fc2': tf.Variable(bout)
    }
    return (weights, biases)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

if (Test):
    weights,biases = initialize_variables()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
if (Test):
    W_conv1 = weights['cov1']
    b_conv1 = biases['cov1']
else:
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

if (Test):
    W_conv2 = weights['cov2']
    b_conv2 = biases['cov2']
else:
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

if (Test):
    W_fc1 = weights['fc1']
    b_fc1 = biases['fc1']
else:
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

if (Test):
    W_fc2 = weights['fc2']
    b_fc2 = biases['fc2']
else:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

if (Test):
    # check how many weights are prunned
    print('-'*79)
    print('Pruning information')
    total_weights_cnt = 0
    (non_zeros, total) = calculate_non_zero_weights(W_conv1.eval())
    total_weights_cnt += total
    print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(W_conv2.eval())
    total_weights_cnt += total
    print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(W_fc1.eval())
    total_weights_cnt += total
    print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    (non_zeros, total) = calculate_non_zero_weights(W_fc2.eval())
    total_weights_cnt += total
    print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    total_weights_cnt += len(b_conv1.eval().flatten())
    total_weights_cnt += len(b_conv2.eval().flatten())
    total_weights_cnt += len(b_fc1.eval().flatten())
    total_weights_cnt += len(b_fc2.eval().flatten())
    print('total number of weights: is now: {}'.format(total_weights_cnt))
else:
    # train the model
    for i in range(200000):
    # for i in range(2000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

predict = y_conv.eval({x: mnist.test.images[:64],
                                    y_: mnist.test.labels[:64],
                                    keep_prob: 1.0})
print('-'*79)
print("View the contents in conv1 layer")
print(W_conv1.eval())
real = mnist.test.labels[:64]
print('-'*79)
print("Neural Network's preditiion of 64 images in the test set")
print(np.argmax(predict, axis = 1))
print('-'*79)
print("Actual value of 64 images in the test set")
print(np.argmax(real, axis = 1))
print('-'*79)
print("test accuracy %.5f"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
if (MASK_GEN == True):
    print("generating the mask pickle file")
    mask_gen()
if (Test == False):
    print("store weights and biases")
    with open('weights.pkl','wb') as f:
        pickle.dump((
            W_conv1.eval(),
            W_conv2.eval(),
            W_fc1.eval(),
            W_fc2.eval(),
            b_conv1.eval(),
            b_conv2.eval(),
            b_fc1.eval(),
            b_fc2.eval(),
        ),f)
