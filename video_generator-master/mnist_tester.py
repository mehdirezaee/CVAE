import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


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
		if (counter_six<samplesNum and np.flatnonzero(mnist.train.labels[k])[0]==np.flatnonzero(mnist.train.labels[2])[0]):
			counter_six=counter_six+1
			data_mem_six=np.concatenate([data_mem_six,[mnist.train.images[k,:]]],axis=0)

		if (counter_three<samplesNum and np.flatnonzero(mnist.train.labels[k])[0]==np.flatnonzero(mnist.train.labels[8])[0]):
			counter_three=counter_three+1
			data_mem_three=np.concatenate([data_mem_three,[mnist.train.images[k,:]]],axis=0)

print('data_mem_three',data_mem_three.shape)
print('data_mem_six',data_mem_six.shape)

def frey_next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network_enc weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


rawData=np.empty((0,size_subj))
rawData=np.concatenate([rawData,data_mem_three],axis=0)
rawData=np.concatenate([rawData,data_mem_six],axis=0)
print('rawData',rawData.shape)
w,n_samples=rawData.shape
network_enc={
'n_hidden_recog_1':200, # 1st layer encoder neurons
'n_hidden_recog_2':200, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_input':784, # MNIST data input (img shape: 28*28)
'n_z':20,
'batch_size':20
}
network_dec={
'n_hidden_gen_1':200, # 1st layer encoder neurons
'n_hidden_gen_2':200, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_hidden_gener_2':200, # 2nd layer decoder neurons # MNIST data input (img shape: 28*28)
'n_input':784, # MNIST data input (img shape: 28*28)
'n_z':20,
'batch_size':20
}
####################### Encoder #############################
x = tf.placeholder(dtype=tf.float32, shape=[None, network_enc['n_input']])

x_h1_w=tf.Variable(xavier_init(network_enc['n_input'], network_enc['n_hidden_recog_1']))
x_h1_b=tf.Variable(tf.zeros(network_enc['n_hidden_recog_1'], dtype=tf.float32))

h1_h2_w=tf.Variable(xavier_init(network_enc['n_hidden_recog_1'],  network_enc['n_hidden_recog_2']))
h1_h2_b=tf.Variable(tf.zeros([ network_enc['n_hidden_recog_2']], dtype=tf.float32))

h2_z_w_mean=tf.Variable(xavier_init( network_enc['n_hidden_recog_2'],  network_enc['n_z']))
h2_z_b_mean=tf.Variable(tf.zeros( network_enc['n_z'], dtype=tf.float32))

h2_z_w_var=tf.Variable(xavier_init( network_enc['n_hidden_recog_2'],  network_enc['n_z']))
h2_z_b_var=tf.Variable(tf.zeros(network_enc['n_z'], dtype=tf.float32))

######################## Z layer ############################
layer_1=tf.nn.softplus(tf.add(tf.matmul(x,x_h1_w),x_h1_b))
layer_2=tf.nn.softplus(tf.add(tf.matmul(layer_1,h1_h2_w),h1_h2_b))

z_mean=tf.add(tf.matmul(layer_2,h2_z_w_mean),h2_z_b_mean)
z_log_sigma_sq=tf.add(tf.matmul(layer_2,h2_z_w_var),h2_z_b_var)

####################### decoder #############################
z_h1r_w=tf.Variable(xavier_init(network_dec['n_z'], network_dec['n_hidden_gen_1']))
z_h1r_b=tf.Variable(tf.zeros(network_dec['n_hidden_gen_1'], dtype=tf.float32))

h1r_h2r_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_1'],network_dec['n_hidden_gen_2']))
h1r_h2r_b=tf.Variable(tf.zeros(network_dec['n_hidden_gen_2']))

h2_recons_mean_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_2'],  network_dec['n_input']))
h2_reocns_mean_b=tf.Variable(tf.zeros(network_dec['n_input']))

h2_recons_var_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_2'],  network_dec['n_input']))
h2_reocns_var_b=tf.Variable(tf.zeros(network_dec['n_input']))
################## Building Z #############################

eps=tf.random_normal((network_enc['batch_size'],network_enc['n_z']),0,1,dtype=tf.float32)
z=tf.add(z_mean,tf.matmul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
#################### Output ###############################
layer_1_recons=tf.nn.softplus(tf.add(tf.matmul(z,z_h1r_w),z_h1r_b))
layer_2_recons=tf.nn.softplus(tf.add(tf.matmul(layer_1_recons,h1r_h2r_w),h1r_h2r_b))
x_reconstr_mean= tf.sigmoid(tf.add(tf.matmul(layer_2_recons,h2_recons_mean_w),h2_reocns_mean_b))
x_reconstr_log_sigma=tf.add(tf.matmul(layer_2_recons,h2_recons_var_w),h2_reocns_var_b)
################## Loss ##################################
# reconstr_loss = \
# -tf.reduce_sum(-0.5*tf.log(2*np.pi)-0.5*(x_reconstr_log_sigma)-tf.divide(
# 	tf.pow((x-x_reconstr_mean),2),2*tf.exp(x_reconstr_log_sigma))
#                            ,1)

reconstr_loss = \
    -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean)
                   + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean),
                   1)

latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
	- tf.square(z_mean)-tf.exp(z_log_sigma_sq), 1)

cost=tf.reduce_mean(reconstr_loss + latent_loss) 
################ Optimizer ###############################
optimizer = \
tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
################## Train   ###############################
training_epochs=5000
display_step=100
plt.ion()
fig=plt.figure()
for k in range(0,1):
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    Train_size=0.7
    control_size=samplesNum
    patient_size=samplesNum

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

    groups=np.concatenate((-1*np.ones((1,int(round(0.7*samplesNum)))),np.ones((1,int(round(0.7*samplesNum))))),axis=1)
    #True_class=np.concatenate((-1*np.ones((1,30)),np.ones((1,30))),axis=1)
    True_class=np.concatenate((-1*np.ones((1,int(round(0.3*samplesNum)))),np.ones((1,int(round(0.3*samplesNum))))),axis=1)

    print('X_train',X_train.shape)
    print('X_test',X_test.shape)

    for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / network_enc['batch_size'])
            #plt.figure('image')
            #plt.imshow(vae.my_reconstr_show_mean.reshape(28, 20), cmap='gray')
            #plt.show(block=False)
            # Loop over all batches
            for i in range(total_batch):
                #batch_xs, _ = mnist.train.next_batch(batch_size)
                frey_images=X_train
                batch_xs = frey_next_batch(network_enc['batch_size'],frey_images)
                # Fit training using batch data
                cost_show,my_reonstr,_=sess.run((cost,x_reconstr_mean,optimizer),feed_dict={x:batch_xs})
                avg_cost += cost_show / n_samples * network_enc['batch_size']
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                    "cost=", "{:.9f}".format(avg_cost)) 
                plt.subplot(121)
                plt.imshow(my_reonstr[0, :].reshape(28, 28), cmap='gray')

                plt.subplot(122)
                plt.imshow(batch_xs[0, :].reshape(28, 28), cmap='gray')
                plt.pause(0.001)
    # z_mean_train=sess.run(z_mean,feed_dict={x:X_train})
    # print('z_mean_train',z_mean_train.shape)
    # z_mean_test=sess.run(z_mean,feed_dict={x:X_test})
    # clf = svm.SVC(kernel='poly',degree=4)
    # clf.fit(z_mean_train,np.ravel(groups))
    # test_predict=np.ravel(clf.predict(z_mean_test))
    # test_classification_rate=(np.count_nonzero(True_class-test_predict)/True_class.shape[1])
    # print('poly_rate',1-test_classification_rate)
    # print('--------------------------------')

#plt.ion()





