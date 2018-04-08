import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as spio
from matplotlib import animation, rc

rawData = spio.loadmat('aod13d.mat', squeeze_me=True)
rawData=rawData['dat3f']

raw_min=np.min(rawData)
raw_max=np.max(rawData)
rawData=(rawData-raw_min)/(raw_max-raw_min)
print(np.min(rawData))
#fig=plt.figure()
#ims=[]
'''
for i in range(0,45):
	im=plt.imshow(rawData[20,:,:,i],cmap='gray',animated=True)
	ims.append([im])
	#plt.show()

#anim = animation.ArtistAnimation(fig, ims, interval=150, blit=True,repeat_delay=1000)
#anim.save('dynamic_images.mp4')
'''