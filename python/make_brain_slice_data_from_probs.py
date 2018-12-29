import numpy as np # linear algebra
from Images import brain as br
from Images import im_tools as imt
import os
import IPython
from collections import Counter
import matplotlib.pyplot as plt

# initialize
#bnames=['Brats17_TCIA_620_1','Brats17_TCIA_621_1','Brats17_TCIA_623_1','Brats17_TCIA_624_1','Brats17_TCIA_625_1']
#bnames = ['Brats18_TCIA01_201_1', 'Brats18_TCIA02_374_1', 'Brats18_TCIA06_165_1']
#bnames = ['Brats18_2013_19_1']
#odir = '/Users/sameer/Documents/research/kacrf/data/lgbm_data/'
#vdir = '/Users/sameer/Documents/research/kacrf/data/lgbm_pics/'

# edit these -- can search either model dir or brain dir
mdir_top = '/Users/sameer/Documents/data/dnn18/HGG/'
bdir_top = '/Users/sameer/Documents/data/training18/HGG'

# output directories, v: problem figure, o: data. None saves nothing
vdir = None
odir = None

# Search mdir for brain names
bnames = os.listdir( mdir_top )
#bnames = os.listdir( bdir_top ) # only needed on henry


for bname in bnames:
    bdir = os.path.join(bdir_top,bname)
    mdir = os.path.join(mdir_top,bname)
    
    # if were
    if not os.path.isdir( mdir ) or bname.startswith('.'):
        print('skipping the following path since not directory: ' + mdir)
        continue


    # create brain, read seg
    bb = br.BrainReader(bdir,bname)
    bp = br.ProbReader(mdir,bname)

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
    bsp = br.BrainSliceProblem.CreateBSPFromBaseProbs(bb,bp,slc1)
    bsp.View(sdir = vdir)
    bsp.Save(sdir = odir) 

    # compute dice for output
    sg = bsp.seg
    sg[sg != 0] = 1
    d1 = imt.BinaryDiceFromProbability(bsp.probs,sg)
    
    # initialize bsp slc 2 and view
    bsp = br.BrainSliceProblem.CreateBSPFromBaseProbs(bb,bp,slc2)
    bsp.View(sdir = vdir)
    bsp.Save(sdir = odir) 

    ## compute dice for output
    sg = bsp.seg
    sg[sg != 0] = 1
    d2 = imt.BinaryDiceFromProbability(bsp.probs,sg)
    
    print("Brain: ", bname, " mstr: Shashank probs")
    print(" d_tum (",slc1, " ): ", d1, " d_half(",slc2, " ): ", d2)


plt.show()


