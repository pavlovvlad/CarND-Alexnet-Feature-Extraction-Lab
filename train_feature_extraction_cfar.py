import time
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

from keras.datasets import cifar10
(X_train, y_train), (X_valid, y_valid) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_valid = y_valid.reshape(-1)

# TODO: Load traffic signs data.
#training_file = 'train.p'	
nb_classes = 10
BATCH_SIZE = 128
EPOCHS = 15

#with open(training_file, mode='rb') as f:
#	train = pickle.load(f)

# TODO: Split data into training and validation sets.
#X_train, X_valid, y_train, y_valid = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=5)

# TODO: Define placeholders and resize operation.
img = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(img, (227, 227))

labels = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(labels, nb_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix

# fcn8, 43
fcn8W = tf.Variable(tf.truncated_normal(shape,stddev=0.01))
fcn8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.matmul(fc7, fcn8W) + fcn8b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# Create a training pipeline
rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate) 
training_operation = optimizer.minimize(loss_operation, var_list=[fcn8W, fcn8b])
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={img: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

init = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        t = time.time()
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={img: batch_x, labels: batch_y})
        
        # validation
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Time: %.3f seconds" % (time.time() - t))
        print()
    print("Finished")