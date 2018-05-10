import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
im = np.reshape(mnist.train.images[5,:], [28,28])
# plt.imshow(mnist.train.images[7,:].reshape(28, 28), cmap='gray') 
# plt.show()
num_subj=mnist.train.labels.shape[0]
size_subj=mnist.train.images[0,:].shape[0]
print('size_subj',size_subj)
data_mem_four=np.empty((0,size_subj))
data_mem_nine=np.empty((0,size_subj))

data_label_four=np.empty((0,1))
data_label_nine=np.empty((0,1))

counter_four=0
counter_nine=0
#data_mem=np.array([])
samplesNum=400;
#plt.imshow(mnist.train.images[2,:].reshape(28, 28), cmap='gray') 
### this is number 4
#plt.imshow(mnist.train.images[8,:].reshape(28, 28), cmap='gray')
### this is number 9
# plt.show()
a=np.array([1,2,3,4,5]).reshape(5,1)
b=np.array([4,2,3,4,5]).reshape(5,1)
a=a>3
b=b<3
for k in range(0,5000):
		if (counter_four<samplesNum and np.flatnonzero(mnist.train.labels[k])[0]==np.flatnonzero(mnist.train.labels[0])[0]):
			counter_four=counter_four+1
			data_mem_four=np.concatenate([data_mem_four,[mnist.train.images[k,:]]],axis=0)
			#data_label_four=np.concatenate([data_label_four,0],axis=0)

		if (counter_nine<samplesNum and np.flatnonzero(mnist.train.labels[k])[0]==np.flatnonzero(mnist.train.labels[8])[0]):
			counter_nine=counter_nine+1
			data_mem_nine=np.concatenate([data_mem_nine,[mnist.train.images[k,:]]],axis=0)
			#data_label_nine=np.concatenate([data_label_nine,1],axis=0)


print('data_mem_nine',data_mem_nine.shape)
print('data_mem_four',data_mem_four.shape)
print(mnist.train.labels[2])

data_label_four=np.ones([data_mem_four.shape[0],1])
data_label_nine=np.zeros([data_mem_nine.shape[0],1])

def frey_next_batch(num, data,labels):
    #Return a total of `num` random samples and labels.
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    label_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle),np.asarray(label_shuffle)

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network_enc weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


rawData=np.empty((0,size_subj))
rawData=np.concatenate([rawData,data_mem_four],axis=0)
rawData=np.concatenate([rawData,data_mem_nine],axis=0)
print('rawData',rawData.shape)
w,n_samples=rawData.shape
rawLabel=np.concatenate([data_label_four,data_label_nine])
print('rawLabel',rawLabel.shape)


network_enc={
'n_hidden_recog_1':70, # 1st layer encoder neurons
'n_hidden_recog_2':70, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_input':785, # MNIST data input (img shape: 28*28)
'n_z':20,
'batch_size':20,
'n_data_input':784
}
network_dec={
'n_hidden_gen_1':70, # 1st layer encoder neurons
'n_hidden_gen_2':70, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_hidden_gener_2':70, # 2nd layer decoder neurons # MNIST data input (img shape: 28*28)
'n_input':785, # MNIST data input (img shape: 28*28)
'n_data_input':784,
'n_z':20,
'batch_size':20
}
####################### Encoder #############################
x_data = tf.placeholder(dtype=tf.float32, shape=[None, network_enc['n_data_input']])
x_test=tf.placeholder(dtype=tf.float32, shape=[None, network_enc['n_data_input']])
y_data= tf.placeholder(dtype=tf.float32,shape=[None,1])
#x = tf.placeholder(dtype=tf.float32, shape=[None, network_enc['n_input']])
#x = tf.concat([x_data,y_data],1)
x = x_data

y_h1_w=tf.Variable(xavier_init(1, network_enc['n_hidden_recog_1']))
y_h1_b=tf.Variable(tf.zeros(network_enc['n_hidden_recog_1'], dtype=tf.float32))
x_h1_w=tf.Variable(xavier_init(network_enc['n_data_input'], network_enc['n_hidden_recog_1']))
x_h1_b=tf.Variable(tf.zeros(network_enc['n_hidden_recog_1'], dtype=tf.float32))

h1_h2_w=tf.Variable(xavier_init(network_enc['n_hidden_recog_1'],  network_enc['n_hidden_recog_2']))
h1_h2_b=tf.Variable(tf.zeros([ network_enc['n_hidden_recog_2']], dtype=tf.float32))

h2_z_w_mean=tf.Variable(xavier_init( network_enc['n_hidden_recog_2'],  network_enc['n_z']))
h2_z_b_mean=tf.Variable(tf.zeros( network_enc['n_z'], dtype=tf.float32))

h2_z_w_var=tf.Variable(xavier_init( network_enc['n_hidden_recog_2'],  network_enc['n_z']))
h2_z_b_var=tf.Variable(tf.zeros(network_enc['n_z'], dtype=tf.float32))

######################## Z layer ############################
layer_1=tf.nn.softplus(tf.add(tf.matmul(x,x_h1_w),x_h1_b))+tf.nn.softplus(tf.add(tf.matmul(y_data,y_h1_w),y_h1_b))
layer_2=tf.nn.softplus(tf.add(tf.matmul(layer_1,h1_h2_w),h1_h2_b))

z_mean=tf.add(tf.matmul(layer_2,h2_z_w_mean),h2_z_b_mean)
z_log_sigma_sq=tf.add(tf.matmul(layer_2,h2_z_w_var),h2_z_b_var)


layer_1_test=tf.nn.softplus(tf.add(tf.matmul(x_test,x_h1_w),x_h1_b))
layer_2_test=tf.nn.softplus(tf.add(tf.matmul(layer_1_test,h1_h2_w),h1_h2_b))
z_mean_test=tf.add(tf.matmul(layer_2_test,h2_z_w_mean),h2_z_b_mean)
z_log_sigma_sq_test=tf.add(tf.matmul(layer_2_test,h2_z_w_var),h2_z_b_var)

####################### decoder #############################
z_h1r_w=tf.Variable(xavier_init(network_dec['n_z'], network_dec['n_hidden_gen_1']))
z_h1r_b=tf.Variable(tf.zeros(network_dec['n_hidden_gen_1'], dtype=tf.float32))

h1r_h2r_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_1'],network_dec['n_hidden_gen_2']))
h1r_h2r_b=tf.Variable(tf.zeros(network_dec['n_hidden_gen_2']))

h2_recons_mean_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_2'],  network_dec['n_data_input']))
h2_reocns_mean_b=tf.Variable(tf.zeros(network_dec['n_data_input']))

h2_recons_var_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_2'],  network_dec['n_data_input']))
h2_reocns_var_b=tf.Variable(tf.zeros(network_dec['n_data_input']))

h2_recons_y_w=tf.Variable(xavier_init(network_dec['n_hidden_gen_2'],  1))

################## Building Z #############################

eps=tf.random_normal((network_enc['batch_size'],network_enc['n_z']),0,1,dtype=tf.float32)
z=tf.add(z_mean,tf.matmul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

eps_test=tf.random_normal((network_enc['batch_size'],network_enc['n_z']),0,1,dtype=tf.float32)
z_test=tf.add(z_mean_test,tf.matmul(tf.sqrt(tf.exp(z_log_sigma_sq_test)), eps_test))
#################### Output ###############################
layer_1_recons=tf.nn.softplus(tf.add(tf.matmul(z,z_h1r_w),z_h1r_b))
layer_2_recons=tf.nn.softplus(tf.add(tf.matmul(layer_1_recons,h1r_h2r_w),h1r_h2r_b))
x_reconstr_mean= tf.sigmoid(tf.add(tf.matmul(layer_2_recons,h2_recons_mean_w),h2_reocns_mean_b))
x_reconstr_log_sigma=tf.add(tf.matmul(layer_2_recons,h2_recons_var_w),h2_reocns_var_b)
y_predict=tf.sigmoid(tf.matmul(layer_2_recons,h2_recons_y_w))

layer_1_recons_test=tf.nn.softplus(tf.add(tf.matmul(z_test,z_h1r_w),z_h1r_b))
layer_2_recons_test=tf.nn.softplus(tf.add(tf.matmul(layer_1_recons_test,h1r_h2r_w),h1r_h2r_b))
x_reconstr_mean_test= tf.sigmoid(tf.add(tf.matmul(layer_2_recons_test,h2_recons_mean_w),h2_reocns_mean_b))
x_reconstr_log_sigma_test=tf.add(tf.matmul(layer_2_recons_test,h2_recons_var_w),h2_reocns_var_b)
y_predict_test=tf.sigmoid(tf.matmul(layer_2_recons_test,h2_recons_y_w))
################## Loss ##################################
# reconstr_loss = \
# -tf.reduce_sum(-0.5*tf.log(2*np.pi)-0.5*(x_reconstr_log_sigma)-tf.divide(
# 	tf.pow((x_data-x_reconstr_mean),2),2*tf.exp(x_reconstr_log_sigma))
#                            ,1)

label_reconstr_loss = \
    -tf.reduce_sum(y_data * tf.log(1e-3 + y_predict)
                   + (1-y_data) * tf.log(1e-3 + 1 - y_predict),
                   1)

reconstr_loss = \
    -tf.reduce_sum(x_data * tf.log(1e-8 + x_reconstr_mean)
                   + (1-x_data) * tf.log(1e-8 + 1 - x_reconstr_mean),
                   1)

latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
	- tf.square(z_mean)-tf.exp(z_log_sigma_sq), 1)

cost=tf.reduce_mean(reconstr_loss + latent_loss + label_reconstr_loss) 
################## Run ##################################
################ Optimizer ###############################
optimizer = \
tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
################## Train   ###############################
training_epochs=8000
display_step=100
run=1
if run==1:
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

		X_train_con_label=rawLabel[Index_con_train,:]
		X_train_pat_label=rawLabel[Index_pat_train,:]

		X_train=np.concatenate((X_train_con, X_train_pat), axis=0)
		X_train_label=np.concatenate((X_train_con_label, X_train_pat_label), axis=0)

		X_test_con=rawData[Index_con_test,:]
		X_test_pat=rawData[Index_pat_test,:]
		X_test_con_label=rawLabel[Index_con_test,:]
		X_test_pat_label=rawLabel[Index_pat_test,:]

		X_test=np.concatenate((X_test_con, X_test_pat), axis=0)
		X_test_label=np.concatenate((X_test_con_label, X_test_pat_label), axis=0)

		groups=np.concatenate((-1*np.ones((1,int(round(0.7*samplesNum)))),np.ones((1,int(round(0.7*samplesNum))))),axis=1)
		#True_class=np.concatenate((-1*np.ones((1,30)),np.ones((1,30))),axis=1)
		True_class=np.concatenate((-1*np.ones((1,int(round(0.3*samplesNum)))),np.ones((1,int(round(0.3*samplesNum))))),axis=1)

		batch_xs_data,batch_xs_label = frey_next_batch(network_enc['batch_size'],X_train,X_train_label)
		# print(batch_xs_label)
		for epoch in range(training_epochs):
		        avg_cost = 0.
		        total_batch = int(n_samples / network_enc['batch_size'])
		        #plt.figure('image')
		        #plt.imshow(vae.my_reconstr_show_mean.reshape(28, 20), cmap='gray')
		        #plt.show(block=False)
		        # Loop over all batches
		        for i in range(total_batch):
		            #batch_xs, _ = mnist.train.next_batch(batch_size)
		            batch_xs_data,batch_xs_label = frey_next_batch(network_enc['batch_size'],X_train,X_train_label)
		            # Fit training using batch data
		            cost_show,my_reonstr,y_predict_show,_=sess.run((cost,x_reconstr_mean,y_predict,optimizer),feed_dict={x_data:batch_xs_data,y_data:batch_xs_label})
        		    my_reonstr_mean_test,y_predict_hat_test=sess.run((x_reconstr_mean_test,y_predict_test),feed_dict={x_test:X_test})
		            # print('y_predict_sh',y_predict_show.shape)
		            train_real_label=batch_xs_label.ravel(network_enc['batch_size'],)
		            test_real_label=X_test_label.ravel(X_test.shape[0],)
		            test_predicted_vec=(y_predict_hat_test>0.5).ravel(X_test.shape[0],)
		            train_predicted_vec=(y_predict_show>0.5).ravel(network_enc['batch_size'],)
		            #print(predicted_vec.shape)
		            avg_cost += cost_show / n_samples * network_enc['batch_size']
		        if epoch % display_step == 0:
		            print("Epoch:", '%04d' % (epoch+1),
		                "cost=", "{:.9f}".format(avg_cost)) 
		            print('train_accuracy',sum(train_predicted_vec==train_real_label)/(network_enc['batch_size']))
		            print('test_accuracy',sum(test_predicted_vec==test_real_label)/(test_real_label.shape[0]))
		            print('-----------------------------------')
		            # plt.subplot(121)
		            # plt.imshow(my_reonstr[0, :].reshape(28, 28), cmap='gray')

		            # plt.subplot(122)
		            # plt.imshow(batch_xs_data[0, :].reshape(28, 28), cmap='gray')
		            # plt.pause(0.001)
		       

