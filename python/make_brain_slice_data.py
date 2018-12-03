import numpy as np # linear algebra
from Images import brain as br
from Images import im_tools as imt
import os
import IPython
from collections import Counter

# initialize
bname='Brats17_TCIA_625_1'
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

# save those to file
bb.SaveSliceProblem(sdir=odir,slc=slc1)
bb.SaveSliceProblem(sdir=odir,slc=slc2)
