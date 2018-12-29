import numpy as np # linear algebra
from Images import brain as br
from Images import im_tools as imt
import os
import IPython
from collections import Counter
import matplotlib.pyplot as plt

# initialize
bnames=['Brats17_TCIA_620_1','Brats17_TCIA_621_1','Brats17_TCIA_623_1','Brats17_TCIA_624_1','Brats17_TCIA_625_1']
bnames=['Brats17_TCIA_623_1']
odir = '/Users/sameer/Documents/research/kacrf/ofiles/'
ddir = '/Users/sameer/Documents/research/kacrf/data/'
ddir = None

noises = [0.05,0.10,0.15,0.2,0.25,0.3]
noises = [0.05,0.10]
noises = [1 ,2, 3 ]
smt = 1 

for bname in bnames:
    bdir = os.path.join('/Users/sameer/Documents/data/trainingdata/',bname)
    # create brain, read seg
    bb = br.Brain(bdir,bname)
    seg = bb.ReadSeg()
    
    # find indices of tumor
    i,j,k = np.nonzero(seg)
    
    # get 2 max locs
    cc = Counter(k)
    i1 = 1
    i2 = int( len(cc)/2 )
    slc1 = cc.most_common(i1)[i1-1][0]
    slc2 = cc.most_common(i2)[i2-1][0]
    
    
    # save those to file
    for nz in noises:
        bsp = br.BrainSliceProblem.CreateBSPFromSeg(bb,slc1,segsmooth=nz,smooth = smt )
        pp = bsp.probs
        sg = bsp.seg
    
        bsp.View(sdir = ddir)
        #bsp.Save(sdir = odir)
        #imt.VisImageFromFlat(pp[:,1])
        #imt.VisImageFromFlat(pp[:,1] > 0.5)
    
        sg[sg != 0] = 1
        d1 = imt.BinaryDiceFromProbability(pp,sg)
    
        bsp = br.BrainSliceProblem.CreateBSPFromSeg(bb,slc2,segsmooth=nz,smooth = smt )
        pp = bsp.probs
        sg = bsp.seg
    
        #bsp.View(sdir = ddir)
        #bsp.Save(sdir = odir)
        #imt.VisImageFromFlat(pp[:,1])
        #imt.VisImageFromFlat(pp[:,1] > 0.5)
    
        sg[sg != 0] = 1
        d2 = imt.BinaryDiceFromProbability(pp,sg)
        th,tt,hh = bsp.ComputeMedianIntensityDistances()
    
        print("Brain: ", bname)
        print("seg smooth: ",nz," psmooth: ",smt )
        print(" d_tum (",slc1, " ): ", d1, " d_half(",slc2, " ): ", d2)
        print(" th: " ,th ," tt: ", tt ," hh: ", hh)


plt.show()


