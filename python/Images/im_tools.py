import numpy as np
from sklearn.metrics import f1_score # dice score
from sklearn.preprocessing import normalize # for resetting probabilities


# get dice from unrounded probability matrix (has
def BinaryDiceFromProbability(Q,seg,prob_cut=0.5):
    # round probability 
    qshape = Q.shape
    yg = np.zeros( (qshape[0],1) )
    yg[Q > prob_cut ] = 1.

    # compute dice by calling other func here
    dice = BinaryDice(yg,seg)

    return dice

# method to create image -- move to tools 
def BinaryDice(ytr,yg,target = 1.):
    ytr = ytr.flatten(order='F')
    yg = yg.flatten(order='F')

    return f1_score(ytr,yg,pos_label=target)

# method to normalize probabilities -- can move it to a tools file 
def NormalizeProbabilities(P):
    P = normalize(P, axis=1, norm='l1')
    P = ResetProbabilityZeros(P)
    P = normalize(P, axis=1, norm='l1')
    return P

# method to reset probability zeros to values ok for log -- can move to tools 
def ResetProbabilityZeros(P):
    ep = np.finfo(np.float32).eps
    P[ P < ep ] = ep
    P[ P > (1-ep) ] = 1-ep
    return P

# method to create image -- move to tools 
def CreateProbsFromSeg2D(seg,sigma = 0.35):

    # make probabilities (bg and target)
    probs_bg = np.random.randn( seg.shape[0],seg.shape[1] ) * sigma + 0.25
    probs_tg = np.random.randn( seg.shape[0],seg.shape[1] ) * sigma + 0.75

    # set final array
    probs = np.where(seg != 0, probs_tg, probs_bg)
    probs = ResetProbabilityZeros(probs)
    probs = probs.reshape( (seg.size,1 ),order='F')
    probs = probs.astype(np.float32)
    probs = np.concatenate( (probs,1-probs), axis=1 )

    return probs


# method to create image -- move to tools 
def CreateImage(sz = 20, sigma = 0.3):
    # get background
    seg = np.zeros( (sz,sz) )

    # index of nonzeros
    start_idx = int(2*sz/5)
    mid_idx = int(3*sz/5)
    end_idx = int(4*sz/5)
    target_size = mid_idx - start_idx
    seg[ start_idx:mid_idx, mid_idx:end_idx ] = 1

    # make probabilities and reshape
    probs = np.random.randn( seg.shape[0],seg.shape[1] ) * sigma + 0.25
    probs[ start_idx:mid_idx, mid_idx:end_idx] = np.random.randn( target_size,target_size ) * sigma + 0.75
    probs = ResetProbabilityZeros(probs)
    probs = probs.reshape( (int(sz*sz),1 ),order='F')
    probs = np.concatenate( (probs,1-probs), axis=1 )

    # make image -- N(1,1) -> bg; N(4,1) -> target
    image = np.random.randn( seg.shape[0],seg.shape[1] ) + 1
    image[ start_idx:mid_idx, mid_idx:end_idx ] = np.random.randn( target_size, target_size ) + 4

    return image,probs,seg

