import os
import tensorflow as tf
import numpy as np
import DEN
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
np.set_printoptions(threshold=np.nan)

np.random.seed(1004)
flags = tf.app.flags
flags.DEFINE_integer("max_iter", 4300, "Epoch to train")
flags.DEFINE_float("lr", 0.001, "Learing rate(init) for train")
#flags.DEFINE_integer("batch_size", 256, "The size of batch for 1 iteration")
flags.DEFINE_integer("batch_size", 2, "The size of batch for 1 iteration")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory path to save the checkpoints")
#flags.DEFINE_integer("dims", [784, 312, 128, 10], "Dimensions about layers including output")
flags.DEFINE_integer("dims", [1, 312, 128, 1], "Dimensions about layers including output")
#flags.DEFINE_integer("n_classes", 10, 'The number of classes at each task')
flags.DEFINE_integer("n_classes", 1, 'The number of classes at each task')
flags.DEFINE_float("l1_lambda", 0.00001, "Sparsity for L1")
flags.DEFINE_float("l2_lambda", 0.0001, "L2 lambda")
flags.DEFINE_float("gl_lambda", 0.001, "Group Lasso lambda")
flags.DEFINE_float("regular_lambda", 0.5, "regularization lambda")
flags.DEFINE_integer("ex_k", 10, "The number of units increased in the expansion processing")
flags.DEFINE_float('loss_thr', 0.01, "Threshold of dynamic expansion")
flags.DEFINE_float('spl_thr', 0.05, "Threshold of split and duplication")
FLAGS = flags.FLAGS

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
trainX = mnist.train.images
valX = mnist.validation.images
testX = mnist.test.images

train_labels = mnist.train.labels 
test_labels = mnist.test.labels
val_labels = mnist.validation.labels


trainX = np.matrix('1 2; 3 4')
train_labels = np.matrix('1;0')

testX = np.matrix('1 2; 3 4')
test_labels =  np.matrix('1;0')

valX = np.matrix('1 2; 3 4')
val_labels =  np.matrix('1;0')

data_import = pd.read_csv('400APPLE.csv').values


trainX = data_import[:271,1:2]
testX = data_import[272:345,1:2]
valX = data_import[346:387,1:2]


train_labels = data_import[:271,0:1]
test_labels = data_import[272:345,0:1]
val_labels = data_import[346:387,0:1]


print("TRAIN X")
print(val_labels)
print("END ")

task_permutation = []


for task in range(1):
	task_permutation.append(np.random.permutation(1))

trainXs, valXs, testXs = [], [], []
for task in range(1):
	trainXs.append(trainX[:, task_permutation[task]])
	valXs.append(valX[:, task_permutation[task]])
	testXs.append(testX[:, task_permutation[task]]) 
	

trainXs, valXs, testXs = trainX, valX, testX

print(trainXs)

model = DEN.DEN(FLAGS)
params = dict()
avg_perf = []

#for t in range(FLAGS.n_classes):
for t in range(2):
	#data = (trainXs[t], train_labels, valXs[t], val_labels, testXs[t],test_labels)

	data = (trainXs, train_labels, valXs, val_labels, testXs,test_labels)
	model.sess = tf.Session()
	print("\n\n\tTASK %d TRAINING\n"%(t+1))

	model.T = model.T+1
	model.task_indices.append(t+1)
	model.load_params(params, time = 1)
	perf, sparsity, expansion = model.add_task(t+1, data)

	params = model.get_params()
	model.destroy_graph()
	model.sess.close()

	model.sess= tf.Session()
	print('\n OVERALL EVALUATION')
	model.load_params(params)
	temp_perfs = []
	for j in range(t+1):
		temp_perf = model.predict_perform(j+1, testXs, test_labels)
		temp_perfs.append(temp_perf)
	avg_perf.append( sum(temp_perfs) / float(t+1) )
	print("   [*] avg_perf: %.4f"%avg_perf[t])
	model.destroy_graph()
	model.sess.close()

