import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def make_Q_patches(i):
    row, col = i // 4, i % 4
    bc = [col,3-row]
    codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]

    xs = np.array([bc[0],bc[0],bc[0]+0.5,bc[0]])
    ys = np.array([bc[1],bc[1]+1,bc[1]+0.5,bc[1]])
    verts = []
    for i in range(len(xs)): verts.append((xs[i],ys[i]))
    lpath = Path(verts,codes)

    xs = np.array([bc[0],bc[0]+0.5,bc[0]+1.,bc[0]])
    ys = np.array([bc[1],bc[1]+0.5,bc[1],bc[1]])
    verts = []
    for i in range(len(xs)): verts.append((xs[i],ys[i]))
    dpath = Path(verts,codes)

    xs = np.array([bc[0]+1,bc[0]+0.5,bc[0]+1,bc[0]+1])
    ys = np.array([bc[1],bc[1]+0.5,bc[1]+1,bc[1]])
    verts = []
    for i in range(len(xs)): verts.append((xs[i],ys[i]))
    rpath = Path(verts,codes)

    xs = np.array([bc[0],bc[0]+0.5,bc[0]+1,bc[0]])
    ys = np.array([bc[1]+1,bc[1]+0.5,bc[1]+1,bc[1]+1])
    verts = []
    for i in range(len(xs)): verts.append((xs[i],ys[i]))
    upath = Path(verts,codes)

    return [lpath,dpath,rpath,upath]

def Render_Qtable(ax,obssize,Qtable):

    cmap = cm.viridis
    holes = [5,7,11,12]
    goal = 15
    
    rng = [999.,-999.]
    for i in range(obssize):
        if Qtable[i].min() < rng[0]: rng[0] = Qtable[i].min()
        if Qtable[i].max() > rng[1]: rng[1] = Qtable[i].max()

    for i in range(obssize):
        if i not in holes and i != goal:
            tpaths = make_Q_patches(i)
            for j,p in enumerate(tpaths):
                val = Qtable[i][j]/rng[1]
                patch = patches.PathPatch(p,fc=cmap(val),ec='k')
                ax.add_patch(patch)

        if i in holes:
            row, col = i // 4, i % 4
            bc = [col,3-row]
            codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]

            xs = np.array([bc[0],bc[0]+1,bc[0]+1,bc[0],bc[0]])
            ys = np.array([bc[1],bc[1],bc[1]+1,bc[1]+1,bc[1]])
            verts = []
            for i in range(len(xs)): verts.append((xs[i],ys[i]))
            patch = patches.PathPatch(Path(verts,codes),fc='k',ec='k')
            ax.add_patch(patch)

        if i == goal:
            row, col = i // 4, i % 4
            bc = [col,3-row]
            codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]

            xs = np.array([bc[0],bc[0]+1,bc[0]+1,bc[0],bc[0]])
            ys = np.array([bc[1],bc[1],bc[1]+1,bc[1]+1,bc[1]])
            verts = []
            for i in range(len(xs)): verts.append((xs[i],ys[i]))
            patch = patches.PathPatch(Path(verts,codes),fc='tab:green',ec='k')
            ax.add_patch(patch)
            
    ax.set_xlim(0,4)
    ax.set_ylim(0,4)

def Render_Max(ax,obssize,Qtable):

    cmap = cm.viridis
    holes = [5,7,11,12]
    goal = 15
    
    rng = [999.,-999.]
    for i in range(obssize):
        if Qtable[i].min() < rng[0]: rng[0] = Qtable[i].min()
        if Qtable[i].max() > rng[1]: rng[1] = Qtable[i].max()

    for i in range(obssize):
        if i not in holes and i != goal:
            tpaths = make_Q_patches(i)
            vals = [Qtable[i][_] for _ in range(4)]
            tmx = np.argmax(vals)
            for j,p in enumerate(tpaths):
                val = Qtable[i][j]/rng[1]
                col = 'w'
                if j == tmx:
                    col = 'r'
                patch = patches.PathPatch(p,fc=col,ec='k')
                ax.add_patch(patch)

        if i in holes:
            row, col = i // 4, i % 4
            bc = [col,3-row]
            codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]

            xs = np.array([bc[0],bc[0]+1,bc[0]+1,bc[0],bc[0]])
            ys = np.array([bc[1],bc[1],bc[1]+1,bc[1]+1,bc[1]])
            verts = []
            for i in range(len(xs)): verts.append((xs[i],ys[i]))
            patch = patches.PathPatch(Path(verts,codes),fc='k',ec='k')
            ax.add_patch(patch)

        if i == goal:
            row, col = i // 4, i % 4
            bc = [col,3-row]
            codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY]

            xs = np.array([bc[0],bc[0]+1,bc[0]+1,bc[0],bc[0]])
            ys = np.array([bc[1],bc[1],bc[1]+1,bc[1]+1,bc[1]])
            verts = []
            for i in range(len(xs)): verts.append((xs[i],ys[i]))
            patch = patches.PathPatch(Path(verts,codes),fc='tab:green',ec='k')
            ax.add_patch(patch)
            
    ax.set_xlim(0,4)
    ax.set_ylim(0,4)
