import numpy as np
import os
import nibabel as nib
from . import im_tools as imt


class Brain:
    def __init__(self,_bdir,_bname):
        self.bdir = _bdir
        self.bname = _bname
    
    def __str__(self):
        return "Name: %s \nLoc: %s" % (self.bname, self.bdir)
    
    def ReadT2(self,slc=-1):
        # Set up file
        imname = self.bname + '_t2_normaff.nii.gz'
        impath = os.path.join(self.bdir,imname)
        
        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]
    
    def ReadT1(self,slc=-1):
        # Set up file
        imname = self.bname + '_t1_normaff.nii.gz'
        impath = os.path.join(self.bdir,imname)
        
        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]
    
    def ReadT1ce(self,slc=-1):
        # Set up file
        imname = self.bname + '_t1ce_normaff.nii.gz'
        impath = os.path.join(self.bdir,imname)
        
        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]
    
    def ReadFlair(self,slc=-1):
        # Set up file
        imname = self.bname + '_flair_normaff.nii.gz'
        impath = os.path.join(self.bdir,imname)
        
        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]
    
    def ReadSeg(self,slc=-1):
        # Set up file
        imname = self.bname + '_seg_aff.nii.gz'
        impath = os.path.join(self.bdir,imname)
        
        # load nifti
        nii = nib.load(impath)
        
        # extract numpy array
        if slc == -1:
            return nii.get_data()
        else:
            bla = nii.get_data()
            return bla[:,:,slc]

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

    def SaveSliceProblem(self, sdir =None, slc=-1):
        if sdir is None:
            sdir = self.bdir

        # set up file names
        imname = self.bname + '_s' + str(slc) + '_allmods.bin'
        segname = self.bname + '_s' + str(slc) + '_seg.bin'
        probsname = self.bname + '_s' + str(slc) + '_probs.bin'
    
        # extract all modalities
        t1,t1ce,t2,fl,seg = self.ReadAll(slc)

        # extract probs
        probs = imt.CreateProbsFromSeg2D(seg,sigma = 0.35)

        # reshape each individual array
        t1 = t1.reshape(-1,1,order='F')
        t2 = t2.reshape(-1,1,order='F')
        t1ce = t1ce.reshape(-1,1,order='F')
        fl = fl.reshape(-1,1,order='F')
        seg = seg.reshape(-1,1,order='F')

        # reshape into large array 
        full = np.concatenate([t1,t1ce,t2,fl],axis = 1) 

        # write to binary, so that 4 int at one pixel are contiguous
        impath = os.path.join(sdir,imname)
        fid = open(impath,'bw')
        full.tofile(fid)
        fid.close()

        # write seg to binary 
        impath = os.path.join(sdir,segname)
        fid2 = open(impath,'bw')
        seg.tofile(fid2)
        fid2.close()

        # write probs to binary
        impath = os.path.join(sdir,probsname)
        fid3 = open(impath,'bw')
        probs.tofile(fid3)
        fid3.close()

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

        # write seg to binary 
        impath = os.path.join(sdir,segname)
        seg = np.fromfile(impath,dtype=np.uint8)
        seg = seg.reshape( (240, 240), order='F')

        # write probs to binary
        impath = os.path.join(sdir,probsname)
        probs = np.fromfile(impath,dtype=np.float32)
        probs = probs.reshape( (-1, 240*240), order = 'F')

        return im, seg, probs
