import numpy as np # linear algebra
from Images import brain as br
from Images import im_tools as imt
import os
import IPython
from collections import Counter
import matplotlib.pyplot as plt

# initialize
bnames=['Brats17_TCIA_620_1','Brats17_TCIA_621_1','Brats17_TCIA_623_1','Brats17_TCIA_624_1','Brats17_TCIA_625_1']
odir = '/Users/sameer/Documents/research/kacrf/data/lgbm_data/'
vdir = '/Users/sameer/Documents/research/kacrf/data/lgbm_pics/'
mdir = '/Users/sameer/Documents/data/lgbm_model/'
mstr = '.lgbm.WT.nii.gz'

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
    
    # initialize bsp and view
    bsp = br.BrainSliceProblem.CreateBSPFromFile(bb.bdir,bb.bname,slc1,mstr,mdir = mdir )
    bsp.View(sdir = vdir)
    bsp.Save(sdir = odir) 

    # compute dice for output
    sg = bsp.seg
    sg[sg != 0] = 1
    d1 = imt.BinaryDiceFromProbability(bsp.probs,sg)
    
    # initialize bsp slc 2 and view
    bsp = br.BrainSliceProblem.CreateBSPFromFile(bb.bdir,bb.bname,slc2,mstr,mdir = mdir )
    bsp.View(sdir = vdir)
    bsp.Save(sdir = odir) 

    # compute dice for output
    sg = bsp.seg
    sg[sg != 0] = 1
    d2 = imt.BinaryDiceFromProbability(bsp.probs,sg)
    
    print("Brain: ", bname, " mstr: ",mstr)
    print(" d_tum (",slc1, " ): ", d1, " d_half(",slc2, " ): ", d2)


plt.show()


#bb.SaveSliceProblem(sdir=odir,slc=slc2)
