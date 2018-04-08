import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import scipy.io as spio
'''
Author: Mohammad Mehdi Rezaee Taghiabadi
'''
def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network_enc weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def conv2d(x, in_channels, output_channels, name, reuse = False):
	'''Convolutional Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [5, 5, in_channels, output_channels], initializer = tf.random_uniform_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_channels], initializer = tf.constant_initializer(0.1))

		conv = tf.nn.conv2d(x, w, strides = [1,3,3,1], padding = 'SAME') + b
		return conv

def deconv2d(x, output_shape, name, reuse = False):
	'''Deconvolutional Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [5, 5, output_shape[-1], int(x.get_shape()[-1])], initializer = tf.random_uniform_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_shape[-1]], initializer = tf.constant_initializer(0.1))

		deconv = tf.nn.conv2d_transpose(x, w, output_shape = output_shape, strides = [1,3,3,1]) + b
		return deconv

def dense(x, input_dim, output_dim, name, reuse = False):
	'''Fully-connected Layer'''
	with tf.variable_scope(name, reuse = reuse):
		w = tf.get_variable('w', [input_dim, output_dim], initializer = tf.random_uniform_initializer(stddev = 0.1))
		b = tf.get_variable('b', [output_dim], initializer = tf.constant_initializer(0.1))

		return tf.matmul(x, w) + b

def aod_next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    #print(idx)
    return np.asarray(data_shuffle)

rawData = spio.loadmat('aod13d.mat', squeeze_me=True)
rawData=rawData['dat3f']
raw_min=np.min(rawData)
raw_max=np.max(rawData)
rawData=(rawData-raw_min)/(raw_max-raw_min)
#np.random.seed(0)
#tf.set_random_seed(0)
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
test_mat = spio.loadmat('aod13d.mat', squeeze_me=True)
aod_images=test_mat['dat3f']
aod_images=(aod_images-np.min(aod_images))/(np.max(aod_images)-np.min(aod_images))
aod_images=np.transpose(aod_images)
#aod_images=np.reshape(aod_images,[-1,28,20,1])
n_samples=aod_images.shape[0]
'''

# plt.imshow(aod_images[5,:,:,0],cmap='gray')
# plt.show()
#print(n_samples)
#mm=aod_next_batch(50,aod_images)


'''
im=aod_images[200,:,:]
plt.imshow(aod_images[500,:].reshape(28,20),cmap='gray')
plt.show()
'''
#print('aod_images',aod_images.shape)

batch_size=30
n_samples=rawData.shape[0]
#im=np.asarray(plt.imread('akg.jpg'))
width,height=rawData.shape[1],rawData.shape[2]
depth=rawData.shape[3]

network_enc={
'output1':12, # 1st layer encoder neurons
'output2':5, # 2nd layer encoder neuronsn_hidden_gener_1=200, # 1st layer decoder neurons
'n_input':560, # MNIST data input (img shape: 28*28)
'n_z':20,
'batch_size':30
}

# inp=tf.placeholder(shape=[width,height,depth],dtype=tf.float32)
# x=tf.reshape(inp,shape=[batch_size,width,height,depth])
x=tf.placeholder(shape=[batch_size,width,height,depth],dtype=tf.float32)
print('x',x.get_shape().as_list())

x_h1_w=tf.Variable(tf.random_uniform(shape=[5,5,depth,network_enc['output1']],dtype=tf.float32),name='x_h1_w')
print('x_h1_w',x_h1_w.get_shape().as_list())

x_h1_b=tf.Variable(tf.zeros(network_enc['output1'], dtype=tf.float32),name='x_h1_b')

h1_h2_w=tf.Variable(tf.random_uniform(shape=[5,5,network_enc['output1'],network_enc['output2']],dtype=tf.float32),name='h1_h2_w')
print('h1_h2_w',h1_h2_w.get_shape().as_list())

h1_h2_b=tf.Variable(tf.zeros(network_enc['output2'], dtype=tf.float32),name='h1_h2_b')


y1_enc=tf.nn.softmax(tf.add(tf.nn.conv2d(x,x_h1_w,strides=[1,3,3,1],padding='SAME'),x_h1_b),name='y1_enc')
print('y1_enc',y1_enc.get_shape().as_list())


y2_enc=tf.nn.softmax(tf.add(tf.nn.conv2d(y1_enc,h1_h2_w,strides=[1,3,3,1],padding='SAME'),h1_h2_b),name='y2_enc')
print('y2_enc',y2_enc.get_shape().as_list())

'''
sess=tf.Session()
sess.run(tf.global_variables_initializer())
result=sess.run(y2_enc,feed_dict={x:aod_images[0:30,:]})
print(result)
'''
y2_enc_shape=y2_enc.get_shape().as_list()
y2_enc_flat=tf.reshape(y2_enc,shape=[-1,y2_enc_shape[1]*y2_enc_shape[2]*y2_enc_shape[3]])

y2_enc_z_mean_w=tf.Variable(tf.random_uniform(shape=[y2_enc_flat.get_shape().as_list()[1],network_enc['n_z']]),name='y2_enc_z_mean_w')
print('y2_enc_z_mean_w',y2_enc_z_mean_w.get_shape().as_list())

y2_enc_z_mean_b=tf.Variable(tf.zeros(shape=[network_enc['n_z']]),name='y2_enc_z_mean_b')

y2_enc_z_var_w=tf.Variable(tf.random_uniform(shape=[y2_enc_flat.get_shape().as_list()[1],network_enc['n_z']]),name='y2_enc_z_var_w')
y2_enc_z_var_b=tf.Variable(tf.zeros(shape=[network_enc['n_z']]),name='y2_enc_z_var_b')

z_mean=tf.add(tf.matmul(y2_enc_flat,y2_enc_z_mean_w),y2_enc_z_mean_b,name='z_mean')
print('z_mean',z_mean.get_shape().as_list())

z_log_sigma_sq=tf.add(tf.matmul(y2_enc_flat,y2_enc_z_var_w),y2_enc_z_var_b,name='z_log_sigma_sq')

samples = tf.random_normal([network_enc['batch_size'], network_enc['n_z']], 0, 1, dtype = tf.float32,name='samples')
print('samples',samples.get_shape().as_list())

z = z_mean + (tf.exp(.5*z_log_sigma_sq) * samples)

z_h1_w=tf.Variable(tf.random_uniform(shape=[network_enc['n_z'],y2_enc_flat.get_shape().as_list()[1]]),name='z_h1_w')
print('z_h1_w',z_h1_w.get_shape().as_list())

z_h1_b=tf.Variable(tf.zeros(shape=[y2_enc_flat.get_shape().as_list()[1]]),name='z_h1_b')

_,y2_dec_width,y2_dec_height,y2_dec_filter=y2_enc.get_shape().as_list()
y2_dec=tf.add(tf.matmul(z,z_h1_w),z_h1_b,name='y2_dec')

y2_dec_flat=tf.nn.softmax(tf.reshape(y2_dec,[-1,y2_dec_width,y2_dec_height,y2_dec_filter]))
_,y1_dec_width,y1_dec_height,y1_dec_filter=y1_enc.get_shape().as_list()
print('y2_dec',y2_dec_flat.get_shape().as_list())

y2_dec_h2_w=tf.Variable(tf.random_uniform([5,5,y1_dec_filter,y2_dec_filter]))
print('y2_dec_h2_w',y2_dec_h2_w.get_shape().as_list())

y2_dec_h2_b=tf.Variable(tf.zeros(shape=[network_enc['output1']]))
y1_dec=tf.nn.softmax(tf.add(tf.nn.conv2d_transpose(y2_dec_flat,y2_dec_h2_w,output_shape=y1_enc.get_shape().as_list(),strides = [1,3,3,1]),y2_dec_h2_b))
print('y1_dec',y1_dec.get_shape().as_list())



y1_dec_x_w_mean=tf.Variable(tf.random_uniform([5,5,depth,y1_dec_filter]))
print('y1_dec_x_w_mean',y1_dec_x_w_mean.get_shape().as_list())

y1_dec_x_b_mean=tf.Variable(tf.zeros(shape=[depth]))

y1_dec_x_w_sigma=tf.Variable(tf.random_uniform([5,5,depth,y1_dec_filter]))

y1_dec_x_b_sigma=tf.Variable(tf.zeros(shape=[depth]))

x_reconst_mean=(tf.add(tf.nn.conv2d_transpose(y1_dec,y1_dec_x_w_mean,output_shape=x.get_shape().as_list(),strides = [1,3,3,1]),y1_dec_x_b_mean))
print('x_reconst_mean',x_reconst_mean.get_shape().as_list())

x_reconst_log_sigma=(tf.add(tf.nn.conv2d_transpose(y1_dec,y1_dec_x_w_sigma,output_shape=x.get_shape().as_list(),strides = [1,3,3,1]),y1_dec_x_b_sigma))
print('x_reconst_log_sigma',x_reconst_log_sigma.get_shape().as_list())


x_reconst_mean_flat=tf.reshape(x_reconst_mean,[-1,width*height*depth])
print('x_reconst_mean_flat',x_reconst_mean_flat.get_shape().as_list())

x_reconst_log_sigma_flat=tf.reshape(x_reconst_log_sigma,[-1,width*height*depth])
x_flat=tf.reshape(x,[-1,width*height*depth])


##################### Optimizer ################################
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
	- tf.square(z_mean)-tf.exp(z_log_sigma_sq), 1)
reconstr_loss = \
-tf.reduce_sum(-0.5*tf.log(2*np.pi)-0.5*(x_reconst_log_sigma_flat)-tf.divide(
	tf.pow((x_flat-x_reconst_mean_flat),2),2*tf.exp(x_reconst_log_sigma_flat))
                           ,1)
cost=tf.reduce_mean(reconstr_loss + latent_loss) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
##############################################################


# batch_size=10
# raw_inp=np.reshape(aod_images[500:510,:],[batch_size,width,height])
# plt.imshow(raw_inp[0,:,:],cmap='gray')
# plt.show()
#inp_reshaped=tf.reshape(raw_inp,[-1,width,height,depth])


sess=tf.Session()
sess.run(tf.global_variables_initializer())
training_epochs=200
display_step=5

plt.ion()
for epoch in range(training_epochs):
	avg_cost = 0
	total_batch = int(n_samples / network_enc['batch_size'])
	#print(total_batch
	for i in range(total_batch):
		batch_xs = aod_next_batch(network_enc['batch_size'],rawData)
		cost_show,my_reonstr,_=sess.run((cost,x_reconst_mean,optimizer),feed_dict={x:batch_xs})
		avg_cost += cost_show / n_samples * network_enc['batch_size']
		#print('epoch:',epoch,' i:',i,avg_cost)
	if (epoch % display_step == 0):
		print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
	if (epoch==191):
		ims=[]
		fig=plt.figure()
		for i in range(0,45):
			im=plt.imshow(my_reonstr[20,:,:,i],cmap='gray',animated=True)
			ims.append([im])
			anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True,repeat_delay=1000)
			anim.save('reconstr_vid.mp4')
		ims=[]
		fig=plt.figure()
		for i in range(0,45):
			im=plt.imshow(batch_xs[20,:,:,i],cmap='gray',animated=True)
			ims.append([im])
			anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True,repeat_delay=1000)
			anim.save('main_vid.mp4')

