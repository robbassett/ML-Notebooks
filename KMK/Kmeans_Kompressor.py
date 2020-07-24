from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.io import loadmat as lm
import numpy as np
import imageio

# A script for making a animation of the iterations in
# a K-means based image compressor. The algorithm has
# been deliberately slowed down as the point is illustrative
# rather than something practical.
# RB - 24 July 2020

class Kompressor(object):

    def __init__(self,X,K=16,xin=128,yin=128):
        # Initiate half the colors to off-black, half to off-white
        vals = np.concatenate((np.random.uniform(0,1,int(K/2)),np.random.uniform(254,255,int(K/2))))
        self.centroids = np.copy(vals)
        for i in range(2): self.centroids=np.vstack((self.centroids,np.copy(vals)))
        self.centroids = self.centroids.T
        
        self.X  = X
        self.C  = np.zeros(X.shape[0])
        self.K  = K
        self.r  = 0
        self.loss = 0
        self.xin,self.yin = xin,yin

    # Assign each pixel to the nearest color
    def assign_points(self):
        for i in range(len(self.C)):
            self.r=np.sqrt(np.sum((self.centroids-self.X[i])*(self.centroids-self.X[i]),axis=1))
            self.C[i] = np.where(self.r == np.min(self.r))[0][0]

    # Shift the colors towards the mean of matched pixels
    def move_centroids(self,lrate=1.):
        for i in range(self.K):
            t = np.where(self.C == i)[0]
            if len(t) > 0:
                true_cent = np.array([np.mean(self.X[:,0][t]),np.mean(self.X[:,1][t]),np.mean(self.X[:,2][t])])
                diff = self.centroids[i].astype(float)-true_cent
                self.centroids[i] -= (diff*lrate)
            else:
                # poor poor unused colors :'(
                pass

    # calculate some kind of "loss", whatever
    def closs(self):
        self.loss = np.sum(self.r)

    # Reconstruct the current image
    def remake_im(self,fnm):
        imout = np.zeros(self.X.shape).astype(np.uint8)
        for i in range(self.X.shape[0]):
            imout[i] = self.centroids[self.C[i].astype(np.uint8)]

        imout = imout.reshape((self.xin,self.yin,3))
        self.im = imout
        imout = Image.fromarray(imout)
        imout.save(f'./frame{fnm}.png')
        
# Load the PNG and remove the alpha layer
in_image = np.asarray(Image.open('jp.png'))
if in_image.shape[2] == 4:
    bt = np.zeros((in_image.shape[0],in_image.shape[1],3))
    bt[:,:,:] = in_image[:,:,:-1]
in_image = np.copy(bt)

# Reshape to X by 3 array
insh = in_image.reshape((in_image.shape[0]*in_image.shape[1],3))
tst  = Kompressor(insh,K=32,xin=in_image.shape[0],yin=in_image.shape[1])

# Initialise some stuff
images = []
xx = np.arange(0,100,1)
lrates = np.exp(xx**(0.4))
lrates = (lrates/np.max(lrates))*0.9
lrates += 0.1

# Iterate! Iterate! Iterate!
for i in range(len(lrates)+25):
    tst.assign_points()

    # Catch because len(lrates) < n_iterations
    try:
        tst.move_centroids(lrate=lrates[i])
    except:
        tst.move_centroids()

    tst.remake_im(0)
    images.append(imageio.imread(f'frame0.png'))


imageio.mimsave('test.gif', images)

