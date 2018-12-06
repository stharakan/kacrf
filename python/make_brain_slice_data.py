import numpy as np # linear algebra
from Images import brain as br
from Images import im_tools as imt
import os
import IPython
from collections import Counter

# initialize
bname='Brats17_TCIA_621_1'
bdir = os.path.join('/Users/sameer/Documents/data/trainingdata/',bname)
odir = '/Users/sameer/Documents/research/kacrf/ofiles/'

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
noises = [0.05,0.10,0.15,0.2,0.25,0.3]
noises = [0.05,0.10]
noises = [0.4]
smt = 1 

# save those to file
for nz in noises:
    #im, sg, pp = bb.SaveSliceProblem(sdir=odir,slc=slc1,noise = nz,smooth = smt)
    bsp = bb.SaveSliceProblem(sdir=odir,slc=slc1,noise = nz,smooth = smt)
    pp = bsp.probs
    sg = bsp.seg

    imt.VisImageFromFlat(pp[:,1])
    imt.VisImageFromFlat(pp[:,1] > 0.5)

    sg[sg != 0] = 1

    print(nz)
    print(imt.BinaryDiceFromProbability(pp,sg))




#bb.SaveSliceProblem(sdir=odir,slc=slc2)
