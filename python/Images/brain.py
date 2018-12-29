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
    
    def ReadBase(self,model_string,slc=-1,mdir=None):
        # Set up file
        imname = model_string
        impath = os.path.join(self.bdir,imname)
        if mdir is not None:
            impath = os.path.join(mdir,imname)

        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]

    def ReadModel(self,model_string,slc=-1,mdir=None):
        # Set up file
        imname = self.bname + model_string
        impath = os.path.join(self.bdir,imname)
        if mdir is not None:
            impath = os.path.join(mdir,imname)

        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]

class BrainReader(Brain):
    def __init__(self,_bdir,_bname,_imflag = '', _sgflag = ''):
        self.bdir = _bdir
        self.bname = _bname
        self.imflag = _imflag
        self.sgflag = _sgflag

    def ReadT2(self,slc=-1):
        return self.ReadModel('_t2' + self.imflag + '.nii.gz',slc)

    def ReadT1(self,slc=-1):
        return self.ReadModel('_t1' + self.imflag + '.nii.gz',slc)
    
    def ReadT1ce(self,slc=-1):
        return self.ReadModel('_t1ce' + self.imflag + '.nii.gz',slc)
    
    def ReadFlair(self,slc=-1):
        return self.ReadModel('_flair' + self.imflag + '.nii.gz',slc)
   
    def ReadSeg(self,slc=-1):
        return self.ReadModel('_seg' + self.imflag + '.nii.gz',slc)

    def ReadAll(self, slc=-1):
        seg = self.ReadSeg(slc) 
        t1 = self.ReadT1(slc) 
        t1c = self.ReadT1ce(slc) 
        fl = self.ReadFlair(slc) 
        t2 = self.ReadT2(slc) 

        return t1, t1c, t2, fl, seg

    def ReadImAsArray(self,slc=-1):
        t1,t1ce,t2,fl,seg = self.ReadAll(slc)
        del seg

        # reshape each individual array
        t1 = t1.reshape(-1,1)
        t2 = t2.reshape(-1,1)
        t1ce = t1ce.reshape(-1,1)
        fl = fl.reshape(-1,1,)
        
        # reshape into large array 
        im = np.concatenate([t1,t1ce,t2,fl],axis = 1) 
        return im


class ProbReader(Brain):
    def ReadNecrotic(self, slc=-1):
        return self.ReadBase('prediction_1.nii.gz',slc)

    def ReadEnhancing(self, slc=-1):
        return self.ReadBase('prediction_4.nii.gz',slc)

    def ReadEdema(self, slc=-1):
        return self.ReadBase('prediction_2.nii.gz',slc)

    def ReadAll(self,slc=-1):
        ne = self.ReadNecrotic(slc)
        ed = self.ReadEdema(slc)
        en = self.ReadEnhancing(slc)

        return ed, en, ne

    def ReadTumor(self,slc=-1):
        ed, en, ne = self.ReadAll(slc)

        # add to get wt probabilities
        return ed + en + ne


class BrainSliceProblem:
    def __init__(self,_bb,_slc,_im,_seg,_probs):
        self.bb = _bb
        self.slc = _slc
        self.im = _im
        self.seg = _seg
        self.probs = _probs
        self.imsz = 240

    @classmethod
    def CreateBSPFromBaseProbs(cls,bb,bp,slc):
        # read image
        im = bb.ReadImAsArray(slc)

        # read seg
        seg = bb.ReadSeg(slc)
        seg = seg.astype(np.float32)

        # read probs
        probs = bp.ReadTumor(slc)
        probs = probs.reshape( (probs.size,1),order='C' )
        probs = np.concatenate( (1-probs,probs), axis=1 )

        # return class object
        return cls(bb,slc,im,seg,probs)
        
    @classmethod
    def CreateBSPFromFile(cls,bb,slc,mstr,mdir):
        # create problem
        t1,t1ce,t2,fl,seg = bb.ReadAll(slc)

        # reshape each individual array
        t1 = t1.reshape(-1,1)
        t2 = t2.reshape(-1,1)
        t1ce = t1ce.reshape(-1,1)
        fl = fl.reshape(-1,1,)
        seg = seg.reshape(-1,1)
        seg = seg.astype(np.float32)
        
        # reshape into large array 
        im = np.concatenate([t1,t1ce,t2,fl],axis = 1) 

        # read probs
        probs = imt.CreateBinaryProbsFromModelFile(bb,slc,mstr,mdir)

        
        # return class object
        return cls(bb,slc,im,seg,probs)


    @classmethod
    def CreateBSPFromSeg(cls,bb,slc,segsmooth=1,smooth=1):
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

    # TODO: Move out of this function -- some type of tools func, just need seg and im from this 
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
