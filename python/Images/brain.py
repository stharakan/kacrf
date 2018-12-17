import numpy as np
import os
import nibabel as nib
from . import im_tools as imt
import scipy.ndimage as ndimage
import scipy.spatial as spt 
import matplotlib.pyplot as plt
import IPython

class Brain:
    def __init__(self,_bdir,_bname):
        self.bdir = _bdir
        self.bname = _bname
    
    def __str__(self):
        return "Name: %s \nLoc: %s" % (self.bname, self.bdir)

    def ReadModel(self,model_string,slc=-1):
        # Set up file
        imname = self.bname + model_string
        impath = os.path.join(self.bdir,imname)

        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]

    
    def ReadT2(self,slc=-1):
        return self.ReadModel('_t2_normaff.nii.gz',slc)

    def ReadT1(self,slc=-1):
        return self.ReadModel('_t1_normaff.nii.gz',slc)
    
    def ReadT1ce(self,slc=-1):
        return self.ReadModel('_t1ce_normaff.nii.gz',slc)
    
    def ReadFlair(self,slc=-1):
        return self.ReadModel('_flair_normaff.nii.gz',slc)
    
    def ReadSeg(self,slc=-1):
        return self.ReadModel('_seg_aff.nii.gz',slc)

    def ReadAll(self, slc=-1):
        seg = self.ReadSeg(slc) 
        t1 = self.ReadT1(slc) 
        t1c = self.ReadT1ce(slc) 
        fl = self.ReadFlair(slc) 
        t2 = self.ReadT2(slc) 

        return t1, t1c, t2, fl, seg
    
    def SaveBrainToBinary(self):
        # set up file name
        imname = self.bname + '_allmods.bin'
        impath = os.path.join(self.bdir,imname)
        fid = open(impath,'bw')

        # extract all modalities
        t1,t1ce,t2,fl,seg = self.ReadAll(slc)

        # extract all modalities
        t1 = t1.reshape(t1.size,1,order='F')
        t2 = t2.reshape(t2.size,1,order='F')
        t1ce = t1ce.reshape(t1ce.size,1,order='F')
        fl = fl.reshape(fl.size,1,order='F')

        # reshape into large array 
        full = np.concatenate([t1,t1ce,t2,fl],axis = 1) 

        # write to binary, so that 4 int at one pixel are contiguous
        full.tofile(fid)

    def ReadBrainFromBinary():
        # set up file name
        imname = self.bname + '_allmods.bin'
        impath = os.path.join(self.bdir,imname)

        # read into array
        am = np.fromfile(impath,dtype=np.float32)
        am = am.reshape( (4,240,240,155), order = 'F')

        return am

    def SaveSliceProblem(self, sdir =None, slc=-1,noise = 0.15,smooth = 2):
        if sdir is None:
            sdir = self.bdir

        # set up file names
        imname = self.bname + '_s' + str(slc) + '_allmods.bin'
        segname = self.bname + '_s' + str(slc) + '_seg.bin'
        probsname = self.bname + '_s' + str(slc) + '_probs.bin'
    
        #bsp = BrainSliceProblem(self.bdir,self.bname,slc,_sig = noise,_smooth = smooth)
        bsp = BrainSliceProblem.CreateBSPFromSeg(self.bdir,self.bname,slc,segsmooth = noise,smooth = smooth)

        bsp.Save(sdir = sdir)
        return bsp
    
    def ReadSliceProblem(self,sdir = None,slc=-1):
        if sdir is None:
            sdir = self.bdir

        # set up file names
        imname = self.bname + '_s' + str(slc) + '_allmods.bin'
        segname = self.bname + '_s' + str(slc) + '_seg.bin'
        probsname = self.bname + '_s' + str(slc) + '_probs.bin'
        
        # write to binary, so that 4 int at one pixel are contiguous
        impath = os.path.join(sdir,imname)
        im = np.fromfile(impath,dtype=np.float32)
        im = im.reshape( (4,240,240),order='F' ) 

        # write seg to binary (as a float? -- don't know c++ importing rn)
        impath = os.path.join(sdir,segname)
        seg = np.fromfile(impath,dtype=np.float32)
        seg = seg.reshape( (240, 240), order='F')

        # write probs to binary
        impath = os.path.join(sdir,probsname)
        probs = np.fromfile(impath,dtype=np.float32)
        probs = probs.reshape( (240*240,-1), order = 'F')

        return im, seg, probs


class BrainSliceProblem:
    def __init__(self,_bb,_slc,_im,_seg,_probs):
        self.bb = _bb
        self.slc = _slc
        self.im = _im
        self.seg = _seg
        self.probs = _probs
        self.imsz = 240

    @classmethod
    def CreateBSPFromSeg(cls,_bdir,_bname,_slc,segsmooth=1,smooth=1):
        # set brain up
        bb = Brain(_bdir,_bname)
        slc = _slc

        # create problem
        t1,t1ce,t2,fl,seg = bb.ReadAll(slc)
        
        # extract probs
        probs = imt.CreateSmoothFlipProbsFromSeg2D(seg,t1=t1,segsmooth = segsmooth,smooth = smooth)
        #probs = imt.CreateSmoothProbsFromSeg2D(seg,t1=t1,sigma = self.sig,smooth = self.smooth)

        # reshape each individual array
        t1 = t1.reshape(-1,1)
        t2 = t2.reshape(-1,1)
        t1ce = t1ce.reshape(-1,1)
        fl = fl.reshape(-1,1,)
        seg = seg.reshape(-1,1)
        seg = seg.astype(np.float32)

        # reshape into large array 
        full = np.concatenate([t1,t1ce,t2,fl],axis = 1) 
        im = full

        # return class object
        return cls(bb,slc,im,seg,probs)
    
    def __str__(self):
        return "Name: %s \nLoc: %s \nSlc: %d" % (self.bname, self.bdir,self.slc)

    def Bname(self):
        return self.bb.bname
    
    def Bdir(self):
        return self.bb.bdir
    
    def SquareIm(self):
        return self.im[:,1].reshape(self.imsz,self.imsz)
    
    def SquareSeg(self):
        return self.seg.reshape(self.imsz,self.imsz)

    def SquareProbs(self):
        return self.probs[:,1].reshape(self.imsz,self.imsz)

    def View(self,sdir = None):
        f, axarr = plt.subplots(2,2)
        #axarr[0,0].imshow( self.SquareIm() , interpolation = 'nearest', cmap = 'gray')
        #axarr[0,0].set_title("T1c im")
        #axarr[0,0].axis('off')


        axarr[0,0].imshow( self.SquareSeg() , interpolation = 'nearest', cmap = 'gray')
        axarr[0,0].set_title("True Seg")
        axarr[0,0].axis('off')

        p1 = self.SquareProbs()
        sg = self.SquareSeg()
        sg[sg !=0 ] = 1
        
        f1 = axarr[0,1].imshow( sg - p1, interpolation = 'nearest')
        axarr[0,1].set_title("Diff")
        axarr[0,1].axis('off')
        f.colorbar(f1,ax=axarr[0,1] )
        
        axarr[1,0].imshow( p1 , interpolation = 'nearest', cmap = 'gray')
        axarr[1,0].set_title("Gen Probs")
        axarr[1,0].axis('off')

        axarr[1,1].imshow( p1 > 0.5 , interpolation = 'nearest', cmap = 'gray')
        axarr[1,1].set_title("Gen Seg")
        axarr[1,1].axis('off')

        if sdir is not None:
            imbase = self.Bname() + '_s' + str(self.slc) + '.png'
            impath = os.path.join(sdir,imbase)
            plt.savefig(impath)

    def ComputeMedianIntensityDistances(self):
        # extract masks
        cur_size = self.seg.size
        tumor_mask = self.seg != 0
        npoints = np.sum(tumor_mask)
        nsub = min(npoints, 200)
        brain_mask = self.im[:,3].reshape(cur_size,1) !=0
        tot = np.arange(self.seg.size).reshape(cur_size,1)

        # get subset indices
        tumor_indices = tot[ tumor_mask ]
        healt_indices = tot[ brain_mask & ~tumor_mask]
        sub_tumor = np.random.permutation(tumor_indices)[:200]
        sub_healt = np.random.permutation(healt_indices)[:200]

        
        # get image subsets
        X_tumor = self.im[sub_tumor,:]
        X_healt = self.im[sub_healt,:]

        # compute distances
        tt = spt.distance.pdist(X_tumor, 'sqeuclidean')
        hh = spt.distance.pdist(X_healt, 'sqeuclidean')
        th = spt.distance.cdist(X_tumor,X_healt, 'sqeuclidean')

        # extract median
        ttf = np.median(tt)
        thf = np.median(th)
        hhf = np.median(hh)

        return np.sqrt(thf), np.sqrt(ttf), np.sqrt(hhf)

    def Save(self, sdir = None):
        if sdir is None:
            sdir = self.Bdir()

        # set up file names
        imname = self.Bname() + '_s' + str(self.slc) + '_allmods.bin'
        segname = self.Bname() + '_s' + str(self.slc) + '_seg.bin'
        probsname = self.Bname() + '_s' + str(self.slc) + '_probs.bin'
        
        # write to binary, so that 4 int at one pixel are contiguous
        impath = os.path.join(sdir,imname)
        fid = open(impath,'bw')
        self.im.tofile(fid)
        fid.close()

        # write seg to binary 
        impath = os.path.join(sdir,segname)
        fid2 = open(impath,'bw')
        self.seg.tofile(fid2)
        fid2.close()

        # write probs to binary
        impath = os.path.join(sdir,probsname)
        fid3 = open(impath,'bw')
        p2 = self.probs.reshape(-1,1,order='F')
        p2.tofile(fid3)
        fid3.close()
