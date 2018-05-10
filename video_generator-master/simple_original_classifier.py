import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn import svm




#np.random.seed(0)
#tf.set_random_seed(0)
# test_mat = spio.loadmat('totalAverage.mat', squeeze_me=True)
# frey_images=test_mat['total_average']
# rawData=frey_images
# print(rawData.shape)
clf = svm.SVC(kernel='poly',degree=8)
# samplesNum=271;

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
im = np.reshape(mnist.train.images[0,:], [28,28])
num_subj=mnist.train.labels.shape[0]
size_subj=mnist.train.images[0,:].shape[0]
data_mem_six=np.empty((0,size_subj))
data_mem_three=np.empty((0,size_subj))
counter_six=0
counter_three=0
#data_mem=np.array([])
samplesNum=100;
# plt.imshow(mnist.train.images[3,:].reshape(28, 28), cmap='gray')
# plt.show()

for k in range(0,3010):
        if (counter_six<samplesNum and np.flatnonzero(mnist.train.labels[k])[0]==np.flatnonzero(mnist.train.labels[1])[0]):
            counter_six=counter_six+1
            data_mem_six=np.concatenate([data_mem_six,[mnist.train.images[k,:]]],axis=0)

        if (counter_three<samplesNum and np.flatnonzero(mnist.train.labels[k])[0]==np.flatnonzero(mnist.train.labels[3])[0]):
            counter_three=counter_three+1
            data_mem_three=np.concatenate([data_mem_three,[mnist.train.images[k,:]]],axis=0)

print('data_mem_three',data_mem_three.shape)
print('data_mem_six',data_mem_six.shape)

rawData=np.empty((0,size_subj))
rawData=np.concatenate([rawData,data_mem_three],axis=0)
rawData=np.concatenate([rawData,data_mem_six],axis=0)
for k in range(0,10):
    # sess=tf.Session()
    # sess.run(tf.global_variables_initializer())
    print('k',k)
    Train_size=0.7
    control_size=100
    patient_size=100

    Con_samp=int(round(Train_size*control_size))
    Pat_samp=int(round(Train_size*patient_size))

    ID_con=np.random.permutation(control_size)
    ID_pat=control_size+np.random.permutation(patient_size)

    Index_con_train = ID_con[0:Con_samp]
    Index_con_test = ID_con[Con_samp+0:control_size]

    Index_pat_train = ID_pat[0:Pat_samp]
    Index_pat_test = ID_pat[Pat_samp+0:patient_size]

    n_samples=rawData.shape[0]

    X_train_con=rawData[Index_con_train,:]
    X_train_pat=rawData[Index_pat_train,:]
    X_train=np.concatenate((X_train_con, X_train_pat), axis=0)

    X_test_con=rawData[Index_con_test,:]
    X_test_pat=rawData[Index_pat_test,:]
    X_test=np.concatenate((X_test_con, X_test_pat), axis=0)

    groups=np.concatenate((-1*np.ones((1,int(round(0.7*control_size)))),np.ones((1,int(round(0.7*patient_size))))),axis=1)
    print('groups',groups.shape)
    #True_class=np.concatenate((-1*np.ones((1,30)),np.ones((1,30))),axis=1)
    True_class=np.concatenate((-1*np.ones((1,int(round(0.3*control_size)))),np.ones((1,int(round(0.3*patient_size))))),axis=1)

    print('X_train',X_train.shape)
    print('X_test',X_test.shape)

    clf.fit(X_train,np.ravel(groups))
    test_predict=np.ravel(clf.predict(X_test))
    test_classification_rate=(np.count_nonzero(True_class-test_predict)/True_class.shape[1])
    print('poly_rate',1-test_classification_rate)
    print('--------------------------------')
