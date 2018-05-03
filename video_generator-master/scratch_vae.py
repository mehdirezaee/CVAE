import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio


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

np.random.seed(0)
tf.set_random_seed(0)
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
test_mat = spio.loadmat('frey.mat', squeeze_me=True)
frey_images=test_mat['ff']
frey_images=(frey_images-np.min(frey_images))/(np.max(frey_images)-np.min(frey_images))
frey_images=np.transpose(frey_images)
w,n_samples=frey_images.shape

network_enc={
'n_hidden_recog_1':200, # 1st layer encoder neurons
'n_hidden_recog_2':200, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_input':560, # MNIST data input (img shape: 28*28)
'n_z':20,
'batch_size':20
}
network_dec={
'n_hidden_gen_1':200, # 1st layer encoder neurons
'n_hidden_gen_2':200, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_hidden_gener_2':200, # 2nd layer decoder neurons # MNIST data input (img shape: 28*28)
'n_input':560, # MNIST data input (img shape: 28*28)
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
x_reconstr_mean=tf.add(tf.matmul(layer_2_recons,h2_recons_mean_w),h2_reocns_mean_b)
x_reconstr_log_sigma=tf.add(tf.matmul(layer_2_recons,h2_recons_var_w),h2_reocns_var_b)
################## Loss ##################################
reconstr_loss = \
-tf.reduce_sum(-0.5*tf.log(2*np.pi)-0.5*(x_reconstr_log_sigma)-tf.divide(
	tf.pow((x-x_reconstr_mean),2),2*tf.exp(x_reconstr_log_sigma))
                           ,1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
	- tf.square(z_mean)-tf.exp(z_log_sigma_sq), 1)

cost=tf.reduce_mean(reconstr_loss + latent_loss) 
################ Optimizer ###############################
optimizer = \
tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
################## Train   ###############################
sess=tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs=100
display_step=5
plt.ion()
for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / network_enc['batch_size'])
        #plt.figure('image')
        #plt.imshow(vae.my_reconstr_show_mean.reshape(28, 20), cmap='gray')
        #plt.show(block=False)
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, _ = mnist.train.next_batch(batch_size)
            batch_xs = frey_next_batch(network_enc['batch_size'],frey_images)
            # Fit training using batch data
            cost_show,my_reonstr,_=sess.run((cost,x_reconstr_mean,optimizer),feed_dict={x:batch_xs})
            avg_cost += cost_show / n_samples * network_enc['batch_size']

	        #plt.subplot(221)
        plt.subplot(221)
        plt.imshow(my_reonstr[0, :].reshape(28, 20), cmap='gray')

        plt.subplot(222)
        plt.imshow(batch_xs[0, :].reshape(28, 20), cmap='gray')

        plt.subplot(223)
        plt.imshow(my_reonstr[1, :].reshape(28, 20), cmap='gray')

        plt.subplot(224)
        plt.imshow(batch_xs[1, :].reshape(28, 20), cmap='gray')                    
        plt.pause(0.05)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))            
            '''
            #plt.close()
            #print('this cost:',cost)
            # Compute average loss
            '''

