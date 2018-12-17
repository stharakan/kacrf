import numpy as np # linear algebra
from Images import brain as br
from Images import im_tools as imt
import os
import IPython
from collections import Counter
import matplotlib.pyplot as plt

# initialize
bnames=['Brats17_TCIA_620_1','Brats17_TCIA_621_1','Brats17_TCIA_623_1','Brats17_TCIA_624_1','Brats17_TCIA_625_1']
bnames=['Brats17_TCIA_620_1','Brats17_TCIA_621_1','Brats17_TCIA_623_1','Brats17_TCIA_624_1','Brats17_TCIA_625_1']
bname = 'Brats17_TCIA_623_1'
odir = '/Users/sameer/Documents/research/kacrf/ofiles/temp'
ddir = '/Users/sameer/Documents/research/kacrf/data/'
qdir = '/Users/sameer/Documents/research/kacrf/data/q_out'
bdir = os.path.join('/Users/sameer/Documents/data/trainingdata/',bname)
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
s1str = "_s" + str(slc1)
s2str = "_s" + str(slc2)


sg1 = seg[:,:,slc1]
sg1[sg1 != 0] = 1
sg2 = seg[:,:,slc2]
sg2[sg2 != 0] = 1

for filename in os.listdir(qdir):
    impath = os.path.join(qdir,filename)
    ppath = os.path.join(qdir,filename[:-3] + 'png')

    # load file
    probs = np.fromfile(impath,dtype=np.float32)
    probs = probs.reshape( (240*240,-1), order = 'F')
    p1 = probs[:,1].reshape(240, 240)

    sg = sg1

    
    if s2str in filename:
        # load s2 instead
        sg = sg2


    # save png
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(p1, interpolation = 'nearest', cmap = 'gray')
    f1 = axarr[1].imshow( sg - p1 , interpolation = 'nearest')
    f.colorbar(f1,ax=axarr[1] )
    plt.savefig(ppath)
    plt.close(f)



#plt.show()


#bb.SaveSliceProblem(sdir=odir,slc=slc2)
