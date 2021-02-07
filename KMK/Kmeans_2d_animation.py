import numpy as np
import matplotlib.pyplot as plt
import imageio

# Loop through each x/y position, measure
# the distance to each cluster centre and
# assign the point to the closest cluster.
# x and y are the arrays of x/y positions,
# and c is a 2d array with cluster x/y
# positions.
def assign_points(x,y,c):
    clust = np.zeros(len(x))
    for i in range(len(x)):
        xd,yd = x[i]-c[:,0],y[i]-c[:,1]
        rs = np.sqrt(xd*xd+yd*yd)
        clust[i] = np.argmin(rs)
    return clust

# For each centroid, calculate the mean
# x and y positions of points assigned to
# them then most the centroid position to
# the calculated position
def move_centroids(x,y,cl,c):
    for i in range(c.shape[0]):
        t = np.where(cl == i)[0]
        c[i] = np.mean(x[t]),np.mean(y[t])

def make_plot(x,y,cl,c):
    F = plt.figure(figsize=(9,4),dpi=150)
    ax = plt.Axes(F,[0.0,0.0,1.0,1.0])
    ax.set_axis_off()
    F.add_axes(ax)
    cs = ['r','g','b']
    for i in range(c.shape[0]):
        t = np.where(cl == i)[0]
        ax.plot(x[t],y[t],'o',c=cs[i])
        ax.plot(c[i][0],c[i][1],'k*',ms=28,mec='k',mew=3)
        ax.plot(c[i][0],c[i][1],'*',ms=20,mfc='w',mec=cs[i],mew=2)
    return F

# Loop through the above functions. First
# assign points, then make the plot, then
# move the centroids. Iterate this n times.
def iterate(x,y,c,n=8):
    images = []
    for i in range(n):
        cl = assign_points(x,y,c)
        F = make_plot(x,y,cl,c)
        move_centroids(x,y,cl,c)
        F.savefig('frame.png')
        J = 8 if i != 0 else 16
        for j in range(J): images.append(imageio.imread(f'frame.png'))
    for j in range(15): images.append(imageio.imread(f'frame.png'))
    imageio.mimsave('test.gif', images)
        
    
# Points are created in 3 random clusters based on a normal
# distribution then concatenated into a single set of points
x1,y1 = np.random.normal(5,1,100),np.random.normal(-1,1.2,100)
x2,y2 = np.random.normal(4,.75,150),np.random.normal(2.5,1,150)
x3,y3 = np.random.normal(1,.5,125),np.random.normal(2,.75,125)
x = np.concatenate((x1,x2,x3))
y = np.concatenate((y1,y2,y3))

K = 3
# Initialise the centroids. Here they are placed clost together
# deliberately to increase the time it takes to converge. This
# is done for illustrative purposes.
centroids = np.random.uniform(2,3,6).reshape(3,2)
iterate(x,y,centroids)
