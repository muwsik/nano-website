# -*- coding: utf-8 -*-
#%%time
from sys import argv
import numpy as np
import pandas as pd
import cv2
import csv
from scipy import ndimage, misc
from skimage import measure
from skimage import morphology
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max
#%matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import os
#from tqdm.notebook import tqdm
#import ipyvolume as ipv
#from ipyfilechooser import FileChooser
import multiprocessing
import time
import math
import random as rnd
import docx

# [ ----- fuctions ------------------------
def LoadBlobs(blobsFName, maxsize, prn=False):
    with open(blobsFName, 'r') as fn:
        reader = csv.reader(fn, delimiter=' ')
        blobs_init = np.array(list(reader), dtype=float)

    if prn:
        print('n blobs init :', len(blobs_init))
    blobs = blobs_init
    for i in range(len(blobs_init)-1,0, -1):
        if blobs[i][2] > maxsize:
            blobs =  np.delete(blobs, (i), axis = 0)
    nblobs = len(blobs)
    if prn:
        print('n blobs <= ', maxsize, ':', len(blobs))
    return blobs
# -----------------------------------------
def show_results(image, blobs, method_name, color):
    """Show images with nanoparticles being detected

    Arguments:
    image -- source grayscale image in the form of ndarray
    blobs -- list or array of areas oСЃСЃupied by the nanoparticle 
            (y, x, r) y and x are coordinates of the center and r - radius 
    method_name -- string with method name to form a capture
    color -- the color of nanoparticle area circles 

    """
    img_ = np.copy(image)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True, sharey=True)

    ax.imshow(img_, cmap='gray')

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
        ax.add_patch(c)
    ax.set_title(method_name + ': ' + str(blobs.shape[0]) + ' particles')       
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()
# ----------------------------------------
def blobs_in_roi(blobs, roi):
    """Check if the center of blob is inside ROI  
    
    Arguments
    blobs -- list or array of areas oСЃСЃupied by the nanoparticle 
            (y, x, r) y and x are coordinates of the center and r - radius    
    roi -- (y,x,h,w)
    
    Return blobs list
    """
    indexes = list(map(lambda blob: int(blob[0]) >= roi[0] \
                                and int(blob[1]) >= roi[1] \
                                and int(blob[0]) < roi[0]+roi[2]  \
                                and int(blob[1]) < roi[1]+roi[3], \
                                    blobs))
    return np.copy(blobs[indexes])
# ----------------------------------------
def findIOU4circle(c1, c2):
    """Finds Jaccard similarity measure for two circles, 
       defined by the coordinates of centers and radii.
       c1=[x1,y1,r1], c2=[x2,y2,r2]  
    """

    d = np.linalg.norm(c1[:2] - c2[:2]) #distance betweem centers
        
    rad1sqr = c1[2] ** 2
    rad2sqr = c2[2] ** 2

    if d == 0:
        # the circle centers are the same
        return min(rad1sqr, rad2sqr)/max(rad1sqr, rad2sqr)

    angle1 = (rad1sqr + d ** 2 - rad2sqr) / (2 * c1[2] * d)
    angle2 = (rad2sqr + d ** 2 - rad1sqr) / (2 * c2[2] * d)

    # check if the circles are overlapping
    if (-1 <= angle1 < 1) or (-1 <= angle2 < 1):
        theta1 = np.arccos(angle1) * 2
        theta2 = np.arccos(angle2) * 2

        area1 = (0.5 * theta2 * rad2sqr) - (0.5 * rad2sqr * np.sin(theta2))
        area2 = (0.5 * theta1 * rad1sqr) - (0.5 * rad1sqr * np.sin(theta1))

        return (area1 + area2)/(np.pi*(rad1sqr+rad2sqr) - area1 - area2)

    elif angle1 < -1 or angle2 < -1:
        # Smaller circle is completely inside the largest circle.
        # Intersection area will be area of smaller circle
        # return area(c1_r), area(c2_r)
        return min(rad1sqr, rad2sqr)/max(rad1sqr, rad2sqr)
    return 0
# ----------------------------------------
def accur_estimation2(blobs_gt, blobs_est, roi, thres=0.5):
    
    blobs_gt = blobs_in_roi(blobs_gt, roi)
    blobs_est = blobs_in_roi(blobs_est, roi)  

    length_gt = blobs_gt.shape[0]
    length_est = blobs_est.shape[0]
        
    iou = np.zeros((length_gt, length_est))
    for i in range(length_gt):
        for j in range(length_est):
            iou[i,j] = findIOU4circle(blobs_gt[i], blobs_est[j])
    
    match = 0
    no_match = 0
    fake = 0
    no_match_index = np.zeros(length_gt,dtype = 'bool')
    match_index = np.zeros(length_gt, dtype='bool')
    
    match_matr = np.zeros((length_gt, length_est), dtype = int)

    for i in range(length_gt):
        if max(iou[i])>=thres:
            imax = np.argmax(iou[i])
            match_matr[i,imax] = 1
            
    no_match_gt_blobs =  blobs_gt[no_match_index]    
    
    fake_index = np.zeros(length_est,dtype = 'bool')
    truedetected_blobs_index = np.zeros(length_est,dtype = 'bool')
    for j in range(length_est):
        if sum(match_matr[:,j])>1: 
            imax = np.argmax(iou[:,j])
            match_matr[:, j] = np.zeros(length_gt, dtype = int)
            match_matr[imax, j] = 1 
        if sum(match_matr[:,j]) == 0:
            fake+=1
            fake_index[j] = True
        else:
            truedetected_blobs_index[j] = True
    fake_blobs = blobs_est[fake_index]
        
    for i in range(length_gt): 
        if sum(match_matr[i,:]) == 0: 
            no_match_index[i] = True
        else:
            match_index[i] = True

    no_match = sum(no_match_index)
    match = sum(sum(match_matr))        
    no_match_gt_blobs =  blobs_gt[no_match_index]
    match_blobs = blobs_gt[match_index]
    truedetected_blobs = blobs_est[truedetected_blobs_index]
    
    return match, no_match, fake, no_match_gt_blobs, fake_blobs, match_blobs, truedetected_blobs
# ----------------------------------------
def accur_radii(blobs_gt, blobs_est, roi, thres=0.25):
    
    blobs_gt = blobs_in_roi(blobs_gt, roi)
    blobs_est = blobs_in_roi(blobs_est, roi)  

    length_gt = blobs_gt.shape[0]
    length_est = blobs_est.shape[0]
        
    iou = np.zeros((length_gt, length_est))
    for i in range(length_gt):
        for j in range(length_est):
            iou[i,j] = findIOU4circle(blobs_gt[i], blobs_est[j])
    
    match = 0
    no_match = 0
    fake = 0
    no_match_index = np.zeros(length_gt,dtype = 'bool')
    
    match_matr = np.zeros((length_gt, length_est), dtype = int)

    for i in range(length_gt):
        if max(iou[i])>=thres:
            imax = np.argmax(iou[i])
            match_matr[i,imax] = 1
            
    no_match_gt_blobs =  blobs_gt[no_match_index]    
    
    fake_index = np.zeros(length_est,dtype = 'bool')
    rdif = np.zeros(length_gt)
    rgt = np.zeros(length_gt)
    rest = np.zeros(length_gt)
    for j in range(length_est):
        if sum(match_matr[:,j])>1: 
            imax = np.argmax(iou[:,j])
            match_matr[:, j] = np.zeros(length_gt, dtype = int)
            match_matr[imax, j] = 1 
    for i in range(length_gt):
        if sum(match_matr[i,:])!=0:
            if sum(match_matr[i,:])>1: 
                printf("WARNING: multiple matching")        
            if sum(match_matr[i,:])==1: 
                jj = np.where(match_matr[i,:]==1)[0]
                j = jj[0]
                #print("Est[",j, "] = ", blobs_est[j], " Gt[", i, "]=", blobs_gt[i])
                rest[i] = blobs_est[j][2]
                rgt[i] = blobs_gt[i][2]
                rdif[i] = blobs_est[j][2] - blobs_gt[i][2]
    ind = rgt!=0
    rest = rest[ind]
    rgt = rgt[ind]
    rdif = rdif[ind]
    rdif_mean = np.mean(rdif)
    rdif_std = np.sqrt(np.var(rdif))
    
    return rdif_mean, rdif_std, rdif, rgt, rest
# ----------------------------------------
def accur_estimation_cv(blobs_gt, blobs_est, roi, thres=0.5):
    length_gt = blobs_gt.shape[0]
    length_est = blobs_est.shape[0]
    
    roi_contour = np.array([
                           [[ roi[0], roi[1] ]],
                           [[ roi[0]+roi[2], roi[1] ]],
                           [[ roi[0]+roi[2], roi[1]+roi[3] ]],
                           [[ roi[0], roi[1]+roi[3]]]
                           ])
    idx = []
    for i in range(length_gt):
         idx.append(cv2.pointPolygonTest(roi_contour, (int(blobs_gt[i,0]),int(blobs_gt[i,1])), False)>=0)
    blobs_gt = blobs_gt[idx]
    
    idx = []
    for i in range(length_est):
         idx.append(cv2.pointPolygonTest(roi_contour, (int(blobs_est[i,0]),int(blobs_est[i,1])), False)>=0)
    blobs_est = blobs_est[idx]    
    
    length_gt = blobs_gt.shape[0]
    length_est = blobs_est.shape[0]

    iou = np.zeros((length_gt, length_est))
    
    for i in range(length_gt):
        for j in range(length_est):
            iou[i,j] = findIOU4circle(blobs_gt[i], blobs_est[j])
    
    match = 0
    no_match = 0
    fake = 0
    
    no_match_index = np.zeros(length_gt,dtype = 'bool')
    for i in range(length_gt):
        if max(iou[i])>=thres:
            match+=1
        else:
            no_match+=1
            no_match_index[i]=True
    
    no_match_gt_blobs =  blobs_gt[no_match_index]       
            
    for j in range(length_est):
        if max(iou[:,j])<thres:
            fake+=1
    return match, no_match, fake, no_match_gt_blobs
# ----------------------------------------
'''def Preprocessing(imgFName, med = False, TopHat = True):

    image = cv2.imread(imgFName)
    height = image.shape[0]
    image = image[:890,:]

    img_init = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#image[:,:,2] # Blue conponent is used
    img2bin = img_init
    
    if med:
        from skimage.filters import median
        footprint =  np.ones((3,3))
        img2bin = median(img2bin, footprint=footprint)

    if TopHat:
        selem =  morphology.disk(4)
        img2bin = morphology.white_tophat(img2bin, selem)
    
    return img2bin, height, img_init'''
# ----------------------------------------
def Preprocessing(imgFName, med = False, TopHat = True):

    image = cv2.imread(imgFName)
    height = image.shape[0]
    image = image[:890,:]

    img_init = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#image[:,:,2] # Blue conponent is used
    img2bin = img_init
    
    if med:
        from skimage.filters import median
        img2bin = median(img2bin, np.ones((3,3)))

    if TopHat:
        selem =  morphology.disk(4)
        img2bin = morphology.white_tophat(img2bin, selem)
    
    return img2bin, height, img_init
# ----------------------------------------
def PreprocessingNoCrop(imgFName, med = False, TopHat = True):
# без обрезки изображения
    image = cv2.imread(imgFName)
    height = image.shape[0]
    #image = image[:890,:]

    img_init = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#image[:,:,2] # Blue conponent is used
    img2bin = img_init
    
    if med:
        from skimage.filters import median
        footprint =  np.ones((3,3))
        img2bin = median(img2bin, footprint=footprint)

    if TopHat:
        selem =  morphology.disk(4)
        img2bin = morphology.white_tophat(img2bin, selem)
    
    return img2bin, height, img_init
# ----------------------------------------

def GetROI(img, gt_blobs):
    roi = np.zeros(4, dtype='int')
    roi[0] = max(0, (gt_blobs[:,0]-gt_blobs[:,2]).min()) 
    roi[1] = max(0, (gt_blobs[:,1]-gt_blobs[:,2]).min())
    roi[2] = min(img.shape[0], (gt_blobs[:,0]+gt_blobs[:,2]).max() - roi[0]+1)
    roi[3] = min(img.shape[1], (gt_blobs[:,1]+gt_blobs[:,2]).max() - roi[1]+1)
    return roi
# ----------------------------------------
#def Visualization(temp_img, data2show, roi, blobs_est, gt_blobs, fake_blobs, no_match_gt_blobs):
def Visualization(temp_img, data2show, blobs_est, roi, FIGSIZE = (9,7), _show=True, _save = False, outfname='', _dpi=300, lsize=100, lwidth = 1):
    #data2show = {'results':1, 'ground truth':1, 'missed gt':1, 'fake est':1, 'large':1}
    # large - to colrize particles, that are lager then lsize
    # show = True - to show on a display
    # save = True - to save in a file outfname
    # _dpi - resolution 
    # lwidth - line width
    
    #temp_img = np.copy(img2bin)
    color = ['blue', 'lime', 'red', 'yellow', 'magenta'] #ground truth, calculated, not found, fake, large
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True, sharey=True)
    #ax = axes#.ravel()

    ax.set_xlim(roi[1],roi[1]+roi[3])
    ax.set_ylim(roi[0]+roi[2],roi[0])

    #print(roi[1], roi[1]+roi[3])
    #print(roi[0]+roi[2], roi[0])

    ax.imshow(temp_img, cmap='gray')

    ax.add_patch(plt.Rectangle((roi[1], roi[0]), roi[3], roi[2], edgecolor = 'pink', fill=False, lw=1))

    if data2show['results']:
        #print("\n estimated:")
        i = 0
        for blob in blobs_est:
            y, x, r = blob
               
            if x>=roi[1] and x<=roi[1]+roi[3] and y>=roi[0] and y<=roi[0]+roi[2]:
                i = i+1
                #print(i,": (", int(round(y)), ",",int(round(x)),";",r,": c0=",optc0s7[int(round(y)),int(round(x))], " crit=",optcrits7[int(round(y)),int(round(x))])        
                if data2show['large'] and r>=lsize:
                    _color = color[4]
                else: 
                    _color = color[1]
                c = plt.Circle((x, y), r, color=_color, linewidth=lwidth, fill=False)
                ax.add_patch(c)
    
    if data2show['ground truth']:
        #print("\n gt:")
        i = 0
        for blob in gt_blobs:
            y, x, r = blob
            i = i+1
            # print(i, ": (", int(round(y)), ",",int(round(x)),";",r,": c0=",optc0s7[int(round(y)),int(round(x))], " crit=",optcrits7[int(round(y)),int(round(x))])
            c = plt.Circle((x, y), r, color=color[0], linewidth=lwidth, fill=False)
            ax.add_patch(c)

    if data2show['missed gt']:
        #print("\n missed gt:")
        i = 0
        for blob in no_match_gt_blobs:
            y, x, r = blob
            i = i+1
            xx = int(round(x))
            yy = int(round(y))

            #print(i, ": (", yy, ",",xx,";",r,": c0=",optc0s7[yy,xx], " crit=",optcrits7[yy,xx])
            #print("(", yy-roi[0], ", ", xx-roi[1], ")")
        
            #print(yy-2, yy+2, xx-2, xx+2)
            #print(optc0s7[yy-2:yy+3, xx-2:xx+3])
            #print(optcrits7[yy-2:yy+3, xx-2:xx+3])
            c = plt.Circle((x, y), r, color=color[2], linewidth=lwidth, fill=False)
            ax.add_patch(c)    
        
    if data2show['fake est']:
        #print("\n fake est:")
        i = 0
        for blob in fake_blobs:
            y, x, r = blob
            i = i+1
            if data2show['large'] and r>=lsize:
            #print(i, ": (", int(round(y)), ",",int(round(x)),";",r,": c0=",optc0s7[int(round(y)),int(round(x))], " crit=",optcrits7[int(round(y)),int(round(x))])
                c = plt.Circle((x, y), r, color=color[3], linewidth=lwidth/2, fill=False)              
            else:
                c = plt.Circle((x, y), r, color=color[3], linewidth=lwidth, fill=False)              
            ax.add_patch(c)    
        
    ax.set_axis_off()

    plt.tight_layout()
    if _show:
        plt.show()
    if _save:
        plt.savefig(outfname, dpi=_dpi, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,transparent=False, bbox_inches='tight', pad_inches=0)                    
# ----------------------------------------
def MakeHelpMatrices(xy2, cs):
    """Р’С‹С‡РёСЃР»РµРЅРёРµ РІСЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹С… РјР°С‚СЂРёС† РґР»СЏ СЂРµС€РµРЅРёСЏ РЎР›РђРЈ РІ РњРќРљ РґР»СЏ
    Р°РїРїСЂРѕРєСЃРёРјР°С†РёРё СЌРєСЃРїРѕРЅРµРЅС‚РѕР№ c0*exp(-c1(x^2+y^2))+c2 
    СЃ С„РёРєСЃРёСЂРѕРІР°РЅРЅС‹РјРё Р·РЅР°С‡РµРЅРёСЏРјРё c1 (РїРѕРґР±РёСЂР°СЋС‚СЃСЏ С‚РѕР»СЊРєРѕ c0 Рё СЃ2)
    wsize - СЂР°Р·РјРµСЂ РјР°С‚СЂРёС†С‹ (СЃРєРѕР»СЊР·СЏС‰РµРіРѕ РѕРєРЅР°), РІРЅСѓС‚СЂРё РєРѕС‚РѕСЂС‹С… РЅСѓР¶РЅРѕ СЃС‡РёС‚Р°С‚СЊ СЌРєСЃРїРѕРЅРµРЅС‚С‹ 
    cs - РІРµРєС‚РѕСЂ Р·РЅР°С‡РµРЅРёР№ РєРѕСЌС„С„РёС†РёРµРЅС‚РѕРІ c1
    """
    wsize = np.shape(xy2)[0]
    hsz = wsize//2 # РјР°РєСЃРёРјР°Р»СЊРЅРѕРµ Рё РјРёРЅРёРјР°Р»СЊРЅРѕРµ Р·РЅР°С‡РµРЅРёРµ РїРѕР»РѕР¶РёС‚РµР»СЊРЅС‹С… Рё РѕС‚СЂРёС†Р°С‚РµР»СЊРЅС‹С… РєРѕРѕСЂРґРёРЅР°С‚
    N = wsize*wsize; # РѕР±С‰РµРµ С‡РёСЃР»Рѕ С‚РѕС‡РµРє 
    
    helpMatrs = {}
    for c1 in cs:
        m = np.exp(-c1*xy2)
        a12 = sum(sum(m));
        a11 = sum(sum(np.array(m)*m)); 
        helpMatrs.update({c1: ((m, a11, a12, N))})
    return helpMatrs
# ----------------------------------------
def MakeHelpMatricesNew(wsize, cs):
    """computation of help matrixes to solve SLAE in MNK for approximation
    by the function c0*exp(-c1(x^2+y^2)) with discrete values of c1
    wsize - size of approximation window
    cs - vector of c1 values
    """
    hsz = wsize // 2  # max and min values of positive and negative coordinate values
    xx = np.array((np.arange(-hsz, hsz+1)))
    xx = xx*xx
    xy2 = np.array([xx]*wsize).T + np.array([xx]*wsize)

    helpMatrs = {}
    for c1 in cs:
        m = np.exp(-c1 * xy2)
        helpMatrs.update({c1: m})
    return helpMatrs, xy2
# ----------------------------------------
def BorderPoints(ppxi,ppyi, hsz, img, sizex, sizey, xy2, wsize):
    wxstart = ppxi-hsz
    if wxstart<0: 
        xleft = np.abs(wxstart) 
        wxstart = 0 
    else:
        xleft = 0 
    wxend = ppxi+hsz
    if wxend>sizex-1: 
        xright = wsize-(wxend-sizex)-1
        wxend = sizex-1
    else:
        xright = wsize
    
    wystart = ppyi-hsz 
    if wystart<0:  
        ytop = np.abs(wystart)
        wystart = 0 
    else:
        ytop = 0  

    wyend = ppyi+hsz 
    if wyend>sizey-1: 
        ybottom = wsize-(wyend-sizey)-1
        wyend = sizey-1
    else:
        ybottom = wsize

    bounds = np.array([xleft, xright-1, ytop, ybottom-1])
    xy2bound = xy2[bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1] 
    subimg = img[wxstart:wxend+1, wystart:wyend+1]
    return bounds, xy2bound, subimg
# ----------------------------------------------------
def GetSubimgByMask(img, cx, cy, mask):
    #cx,cy - coordinates of the center of the subimage in img
    wsize, wsize = np.shape(mask)
    hsz = wsize//2
    subimg = np.zeros((wsize, wsize))
    ppx, ppy = np.where(mask == 1)
    nump = np.shape(ppx)[0]
    for i in range(nump):
        subimg[ppx[i],ppy[i]] = img[cx-hsz+ppx[i], cy-hsz+ppy[i]]
    return subimg
# ----------------------------------------
def BorderMask(ppxi, ppyi, hsz, sizex, sizey, wsize):
    wxstart = ppxi - hsz
    if wxstart < 0:
        xleft = np.abs(wxstart)
        wxstart = 0
    else:
        xleft = 0
    wxend = ppxi + hsz
    if wxend > sizex - 1:
        xright = wsize - (wxend - sizex) - 1
        wxend = sizex - 1
    else:
        xright = wsize

    wystart = ppyi - hsz
    if wystart < 0:
        ytop = np.abs(wystart)
        wystart = 0
    else:
        ytop = 0

    wyend = ppyi + hsz
    if wyend > sizey - 1:
        ybottom = wsize - (wyend - sizey) - 1
        wyend = sizey - 1
    else:
        ybottom = wsize

    mask = np.zeros((wsize,wsize))
    mask[xleft:xright, ytop:ybottom] = 1
    return mask
# ----------------------------------------
def ApproxInWindowMask(wsize, z, c1s, xy2, helpMatrs, mask):
    """Approximation of function z by the exponential function
    c0*exp(-c1(x^2+y^2))
    c1s - vector of c1 values, for each of which the approximation is made (only c0 is determined)
    then the optimal c1 is found
    z is a matrix [wsize x wsize]
    for points near image borders z partially contains zeroes
    helpMatrs is a spacial structure with precomputed matrixes and SLAE coefficients to found с0
    !!! it is supposed that helpMatrs shape and z shape are matched!!!
    mask is a matrix [wsize x wsize] of ones and zeroes. it determines points that participate in approximation
    """

    if (z == 0).all():
        return 0, 0, 0

    hsz = wsize // 2
    npoints = sum(sum(mask))

    nc = np.shape(c1s)
    crits = np.zeros(nc, dtype='float')
    c0s = np.zeros(nc, dtype='float')

    i = 0
    for c1 in c1s:
        # determine the optimal c0 for the given c1
        m = helpMatrs.get(c1)
        m = np.multiply(m, mask)
        a11 = sum(sum(np.array(m) ** 2))
        b1 = sum(sum(z * m))  # ????????
        c0 = b1 / a11

        zapr = np.zeros((wsize, wsize), dtype='float')
        zapr = c0 * np.exp(-c1 * xy2)
        zaprmask = np.multiply(zapr, mask)

        delta = z - zapr
        crit = np.sqrt(sum(sum(np.array(delta) * delta))) / npoints
        crits[i] = crit
        c0s[i] = c0
        i = i + 1

    # optimal c0 and c1 are found
    ind = np.argmin(crits)
    optc0 = c0s[ind]
    optc1 = c1s[ind]
    optcrit = crits[ind]
    return optc0, optc1, optcrit
# ----------------------------------------

def ApproxByExpInWindow(wsize, z, c1s, xy2bound, bounds, helpMatrs, nparams):
    """РђРїРїСЂРѕРєСЃРёРјР°С†РёСЏ РґРІСѓС…РјРµСЃС‚РЅРѕР№ С„СѓРЅРєС†РёРё z СЌРєСЃРїРѕРЅРµРЅС‚РѕР№
    c0*exp(-c1(x^2+y^2))+c2 
    c1s - РІРµРєС‚РѕСЂ Р·РЅР°С‡РµРЅРёР№ c1, РґР»СЏ РєР°Р¶РґРѕРіРѕ РёР· РєРѕС‚РѕСЂС‹С… РїСЂРѕРІРѕРґРёС‚СЃСЏ Р°РїРїСЂРѕРєСЃРёРјР°С†РёСЏ
    СЌРєСЃРїРѕРЅРµРЅС‚РѕР№ (РїРѕРґР±РёСЂР°СЋС‚СЃСЏ С‚РѕР»СЊРєРѕ c0 Рё СЃ2)
    Р·Р°С‚РµРј РІС‹Р±РёСЂР°РµС‚СЃСЏ РѕРїС‚РёРјР°Р»СЊРЅРѕРµ c1 
    helpMatrs - СЃРїРµС†РёР°Р»СЊРЅР°СЏ СЃС‚СЂСѓРєС‚СѓСЂР° СЃ Р·Р°СЂР°РЅРµРµ РІС‹С‡РёСЃР»РµРЅРЅС‹РјРё РјР°С‚СЂРёС†Р°РјРё Рё
    РєРѕСЌС„С„РёС†РёРµРЅС‚Р°РјРё РЎР›РђРЈ РґР»СЏ РЅР°С…РѕР¶РґРµРЅРёСЏ c0 Рё c2
    !!! РїСЂРµРґРїРѕР»Р°РіР°РµС‚СЃСЏ, С‡С‚Рѕ helpMatrs СЃРѕРѕС‚РІРµС‚СЃС‚РІСѓСЋС‚ СЂР°Р·РјРµСЂСѓ z!!!
    z РјРѕР¶РµС‚ Р±С‹С‚СЊ РЅРµ РєРІР°РґСЂР°С‚РЅРѕР№ (РµСЃР»Рё С‚РѕС‡РєР° СЂР°СЃРїРѕР»РѕР¶РµРЅР° Р±Р»РёР·РєРѕ Рє РєСЂР°СЋ)
    bounds - РјР°СЃСЃРёРІ РёР· С‡РµС‚С‹СЂРµС… СЌР»РµРјРµРЅС‚РѕРІ, РѕРіСЂР°РЅРёС‡РёРІР°СЋС‰РёР№ РІСЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹Рµ
    РјР°С‚СЂРёС†С‹ Рё Р°СЂРіСѓРјРµРЅС‚С‹ СЌРєСЃРїРѕРЅРµРЅС‚ РґР»СЏ РєСЂР°РµРІС‹С… С‚РѕС‡РµРє: (xleft, xright, ytop, ybottom) 
    nparams - С‡РёСЃР»Рѕ РѕС†РµРЅРёРІР°РµРјС‹С… РїР°СЂР°РјРµС‚СЂРѕРІ (РїСЂРё nparams=2 РѕС†РµРЅРёРІР°СЋС‚СЃСЏ С‚РѕР»СЊРєРѕ СЃ0 Рё c1, Р° СЃ2 РїСЂРёРЅРёРјР°РµС‚СЃСЏ СЂР°РІРЅС‹Рј 0)
    """
    
    (sizex, sizey) = np.shape(z)
    if (z==0).all(): 
        return 0, 0, 0, 0
    
    hsz = wsize//2
    npoints = sizex*sizey
    nc = np.shape(c1s)
    crits = np.zeros(nc, dtype='float')
    c2s = np.zeros(nc, dtype='float')
    c0s = np.zeros(nc, dtype='float')
      
    i=0
    for c1 in c1s:
        # РЅР°С…РѕРґРёРј РѕРїС‚РёРјР°Р»СЊРЅС‹Рµ c0 Рё СЃ2 РґР»СЏ Р·Р°РґР°РЅРЅРѕРіРѕ c1
        zapr = np.zeros((sizex,sizey), dtype='float')
        (m, a11, a12, a22) = helpMatrs.get(c1)
        if ((sizex<wsize)or(sizey<wsize)): 
            # РєРѕСЂСЂРµРєС‚РёСЂРѕРІРєР° РґР»СЏ РєСЂР°РµРІС‹С… С‚РѕС‡РµРє            
            m = m[bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1] 
            a12 = sum(sum(m))  
            a11 = sum(sum(np.array(m)**2))
            a22 = sizex*sizey  

        if nparams==3:   # РґР»СЏ РѕС†РµРЅРєРё 3-С… РїР°СЂР°РјРµС‚СЂРѕРІ
            b1 = sum(sum(z*m))
            c2 = (sum(sum(z)) - b1)/(a22 - a12)
            c0 = (b1- c2*a12)/a11         
        elif nparams==2: # РґР»СЏ РѕС†РµРЅРєРё 2-С… РїР°СЂР°РјРµС‚СЂРѕРІ
            b1 = sum(sum(z*m))
            c0 = b1/a11 
            c2= 0.
        else: 
            printf('РќРµРєРѕСЂСЂРµРєС‚РЅРѕРµ С‡РёСЃР»Рѕ РѕС†РµРЅРёРІР°РµРјС‹С… РїР°СЂР°РјРµС‚СЂРѕРІ')
            return    
  
        # Р·РЅР°С‡РµРЅРёСЏ Р°РїРїСЂРѕРєСЃРёРјРёСЂСѓСЋС‰РµР№ С„СѓРЅРєС†РёРё
        zapr = c0*np.exp(-c1*xy2bound)+c2

        # РІС‹С‡РёСЃР»СЏРµРј Р·РЅР°С‡РµРЅРёРµ РєСЂРёС‚РµСЂРёСЏ
        delta = z - zapr
        crit = np.sqrt(  sum( sum( np.array(delta)*delta) ) )/npoints
        crits[i] = crit
        c0s[i] = c0
        c2s[i] = c2
        i=i+1
                       
    # РЅР°С…РѕРґРёРј РјРёРЅРёРјР°Р»СЊРЅРѕРµ Р·РЅР°С‡РµРЅРёРµ РєСЂРёС‚РµСЂРёСЏ Рё СЃРѕРѕС‚РІРµС‚СЃС‚РІСѓСЋС‰РёРµ Р·РЅР°С‡РµРЅРёСЏ РїР°СЂР°РјРµС‚СЂРѕРІ
    ind = np.argmin(crits)                       
    optc0 = c0s[ind]
    optc1 = c1s[ind]
    optc2 = c2s[ind]
    optcrit = crits[ind]
    return optc0, optc1, optc2, optcrit  
# ----------------------------------------
def ApproxImageByExpParWorker(args):
# Parallel worker!!!! 
    (ppxi, ppyi, hsz, wsize, subimg, c1s, xy2bound, bounds, helpMatrs, nparams)= args
    c0 = 0
    c1 = 0
    c2 = 0
    crit = 100
    (c0, c1, c2, crit) = ApproxByExpInWindow(wsize, subimg, c1s, xy2bound, bounds, helpMatrs, nparams)
    return ppxi, ppyi, c0, c1, c2, crit
# -------------------------------------------------------
def ParWorkerMask(args):
# Parallel worker!!!!
    (ppxi, ppyi, hsz, wsize, subimg, c1s, xy2, mask, helpMatrs) = args
    c0 = 0
    c1 = 0
    crit = 100
    (c0, c1, crit) = ApproxInWindowMask(wsize, subimg, c1s, xy2, helpMatrs, mask)
    return [ppxi, ppyi, c0, c1, crit]
# ----------------------------------------
def ApproxImageByExp(img, wsize, c1s, nparams, points, nproc):
    """РђРїРїСЂРѕРєСЃРёРјР°С†РёСЏ С„СЂР°РіРјРµРЅС‚РѕРІ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ img, РІС‹СЂРµР·Р°РЅРЅС‹С… РѕРєРЅР°РјРё СЂР°Р·РјРµСЂР° wsize, 
    СЌРєСЃРїРѕРЅРµРЅС‚Р°РјРё z=c0*exp(-c1(xx^2+yy^2))+c2 (c1 РјРѕР¶РµС‚ РїСЂРёРЅРёРјР°С‚СЊ РґРёСЃРєСЂРµС‚РЅС‹Рµ Р·РЅР°С‡РµРЅРёСЏ РёР· РјР°СЃСЃРёРІР° c1s)
    С‚РѕС‡РєРё, Р±Р»РёР·РєРёРµ Рє РєСЂР°СЋ, Р°РїРїСЂРѕРєСЃРёРјРёСЂСѓСЋС‚СЃСЏ РІ СѓСЂРµР·Р°РЅРЅРѕРј РѕРєРЅРµ 
    РѕРєРЅР° Р·Р°РґР°СЋС‚СЃСЏ РєРѕРѕСЂРґРёРЅР°С‚Р°РјРё С†РµРЅС‚СЂРѕРІ (РµРґРёРЅРёС†С‹ РІ РјР°СЃСЃРёРІРµ points) Рё СЂР°Р·РјРµСЂРѕРј (С‚РѕР»СЊРєРѕ РЅРµС‡РµС‚РЅС‹Р№) 
    points - РјР°СЃСЃРёРІ РЅСѓР»РµР№ Рё РµРґРёРЅРёС† 
    points = 'all' (default) - Р°РїРїСЂРѕРєСЃРёРјР°С†РёСЏ РІРѕ РІСЃРµС… С‚РѕС‡РєР°С… 
    """
    mnimg = np.mean(img)
    
    sizex, sizey = np.array(img).shape
    hsz = wsize//2

    # РјР°СЃСЃРёРІС‹ Р·РЅР°С‡РµРЅРёР№ РїР°СЂР°РјРµС‚СЂРѕРІ Рё РєСЂРёС‚РµСЂРёСЏ
    optc0s = np.zeros((sizex, sizey), dtype='float')
    optc1s = np.zeros((sizex, sizey), dtype='float')
    optc2s = np.zeros((sizex, sizey), dtype='float')
    optcrits = np.ones((sizex, sizey), dtype='float')*10 # РІ РЅРµРїРѕСЃС‡РёС‚Р°РЅРЅС‹С… С‚РѕС‡РєР°С… Р±СѓРґРµС‚ 10 !!!
   
    ppx,ppy= np.where(points == 1)
    nump = np.shape(ppx)[0]
    
    # Р·РЅР°С‡РµРЅРёСЏ Р°СЂРіСѓРјРµРЅС‚РѕРІ СЌРєСЃРїРѕРЅРµРЅС‚С‹
    xx = np.array((np.arange(-hsz,hsz+1)))    
    xx = xx*xx
    xy2 = np.array([xx]*wsize).T + np.array([xx]*wsize)
    helpMatrs = MakeHelpMatrices (xy2, c1s)    
        
    #print('Number of points :', nump) 
    reslist = []
    data = []
    for i in range(nump):
        (bounds, xy2bound, subimg) = BorderPoints(ppx[i],ppy[i], hsz, img, sizex, sizey, xy2, wsize)
        temp = (ppx[i], ppy[i], hsz, wsize, subimg, c1s, xy2bound, bounds, helpMatrs, nparams)
        data.append(temp)
    with multiprocessing.Pool(nproc) as pool:
        reslist = pool.map(ApproxImageByExpParWorker, data)
    for i in range(len(reslist)):
        optc0s[reslist[i][0],reslist[i][1]]=reslist[i][2]
        optc1s[reslist[i][0],reslist[i][1]]=reslist[i][3]
        optc2s[reslist[i][0],reslist[i][1]]=reslist[i][4]
        optcrits[reslist[i][0],reslist[i][1]]=reslist[i][5]
        

    return optc0s, optc1s, optc2s, optcrits
# ----------------------------------------
def ApproxImageByExpMask(img, wsize, c1s, points, nproc, maskMain):
    """
    Approximation of img fragments in window of size wsize by exonential function
    z=c0*exp(-c1(xx^2+yy^2))
    (c1 can take only discrete values from the array c1s)
    windows is determned by its centers - points of img
    if window center is near img border, the window contains zeros for over-border points
    points is array of zeros and ones (to consider the point of img as the center of approximation window or not)
    points = 'all' (default) - approximation in all points of img
    """
    mnimg = np.mean(img)

    sizex, sizey = np.array(img).shape
    hsz = wsize // 2
    #optc0s = np.ones((sizex, sizey), dtype='float')*-100
    optc0s = np.zeros((sizex, sizey), dtype='float')
    optc1s = np.zeros((sizex, sizey), dtype='float')
    optcrits = np.ones((sizex, sizey),
                       dtype='float') * 10  # if approximation does not made, the criterion value is 10 !!!

    ppx, ppy = np.where(points == 1)
    nump = np.shape(ppx)[0]

    helpMatrs, xy2 = MakeHelpMatricesNew(wsize, c1s)

    data = []
    for i in range(nump):
        mask = BorderMask(ppx[i], ppy[i], hsz, sizex, sizey, wsize)
        mask = np.multiply(mask, maskMain)
        subimg = GetSubimgByMask(img, ppx[i], ppy[i], mask)
        #if i<10:
        #    print(i, ':', ppx[i], ppy[i], subimg, mask)
        temp = (ppx[i], ppy[i], hsz, wsize, subimg, c1s, xy2, mask, helpMatrs)
        data.append(temp)
    reslist = []
    with multiprocessing.Pool(nproc) as pool:
        reslist = pool.map(ParWorkerMask, data)
        #print(np.shape(reslist))
    for i in range(len(reslist)):
        optc0s[reslist[i][0],reslist[i][1]]=reslist[i][2]
        optc1s[reslist[i][0],reslist[i][1]]=reslist[i][3]
        optcrits[reslist[i][0],reslist[i][1]]=reslist[i][4]

    #with multiprocessing.Pool(nproc) as pool:
    #    for _ppx, _ppy, optc0, optc1, optcrit in pool.map(ParWorkerMask, data):
    #        optc0s[_ppx,_ppy] = optc0
    #        optc1s[_ppx, _ppy] = optc1
    #        optcrits[_ppx, _ppy] = optcrit

    return optc0s, optc1s, optcrits
# ----------------------------------------
def ExponetialApproximation(img, c1s, points, wsize = 7, thresCoefOld = 0.6, nproc = 1):
    # Approximating image fragments with centers in points
    optc0s, optc1s, optc2s, optcrits = ApproxImageByExp(img, wsize, c1s, 2, points, nproc)
   
    # Detecting particles
    particles = peak_local_max(optc0s, min_distance=2, threshold_abs=None, threshold_rel=0.1, footprint=None, labels=None)
    radii = np.zeros((particles.shape[0],1), dtype = 'float')
    optc1s_p = np.zeros((particles.shape[0],1), dtype = 'float')
    sum_optc0 = 0
    for i in range (particles.shape[0]):
        optc1s_p[i] = optc1s[particles[i][0]][particles[i][1]]
        radii[i] = 1/np.sqrt(optc1s_p[i]) 

    blobs_exp = np.hstack((particles, radii))

    # Finding the threshold for filtering (Old version)
    optc1s_p = np.zeros((particles.shape[0],1), dtype = 'float')
    sum_optc0 = 0
    for i in range (particles.shape[0]):
        optc1s_p[i] = optc1s[particles[i][0]][particles[i][1]]
        radii[i] = 1/np.sqrt(optc1s_p[i])
        sum_optc0 = sum_optc0 + optc0s[particles[i][0]][particles[i][1]]
    mean_optc0 = sum_optc0/particles.shape[0]
    thres = mean_optc0*thresCoefOld
    #print("Old thres = ", thres)

    # Filtering
    for i in reversed(range(blobs_exp.shape[0])):   
        if (optc0s[particles[i][0]][particles[i][1]]<thres):  
            blobs_exp = np.delete(blobs_exp, (i), axis=0) 

    return blobs_exp    
# ----------------------------------------
def ExponentialApproximationMask(img, c1s, points, maskMain, wsize=7, thresCoefOld=0.6, nproc=1):
    # Approximation image fragments with centers in points
    # maskMain is matrix [wsize x wsize] of 0 and 1 that defines which points of the window will participate
    # in the approximation or False to take into account all points (it is equivalent to use matrix of ones)
    # wsize - size of window for approximation
    # points - array of cordinates of image points that will consider as centers of windows for approximation
    # thresCoefOld - coefficient to compute adaptive threshold for detection
    # nproc - number of processes
    # c1s - array of values of c1 of function z=c0*exp(-c1(xx^2+yy^2)) for image fragments approximation
    # img - full image
    if (len(np.shape(maskMain)) == 0):
        if (maskMain == False):
            #print('no Mask')
            maskMain = np.ones((wsize, wsize))
        else:
            print('mask should be [wsize x wsize] of False')
    else:
        msx, msy = np.shape(maskMain)
        if ((msx != wsize) or (msy != wsize)):
            print('mask should be [wsize x wsize] of False')

    optc0s, optc1s, optcrits = ApproxImageByExpMask(img, wsize, c1s, points, nproc, maskMain)

    # Detecting particles
    particles = peak_local_max(optc0s, min_distance=2, threshold_abs=None, threshold_rel=0.1, footprint=None,
                               labels=None)
    radii = np.zeros((particles.shape[0], 1), dtype='float')
    optc1s_p = np.zeros((particles.shape[0], 1), dtype='float')
    sum_optc0 = 0
    for i in range(particles.shape[0]):
        optc1s_p[i] = optc1s[particles[i][0]][particles[i][1]]
        radii[i] = 1 / np.sqrt(optc1s_p[i])

    blobs_exp = np.hstack((particles, radii))

    # Finding the threshold for filtering (Old version)
    optc1s_p = np.zeros((particles.shape[0], 1), dtype='float')
    sum_optc0 = 0
    for i in range(particles.shape[0]):
        optc1s_p[i] = optc1s[particles[i][0]][particles[i][1]]
        radii[i] = 1 / np.sqrt(optc1s_p[i])
        sum_optc0 = sum_optc0 + optc0s[particles[i][0]][particles[i][1]]
    mean_optc0 = sum_optc0 / particles.shape[0]
    thres = mean_optc0 * thresCoefOld
    # print("Old thres = ", thres)

    # Filtering
    for i in reversed(range(blobs_exp.shape[0])):
        if (optc0s[particles[i][0]][particles[i][1]] < thres):
            blobs_exp = np.delete(blobs_exp, (i), axis=0)

    return blobs_exp
# ----------------------------------------
def PrintHeader(imgFName, gtFName, methodName, thresCoefOld, wsize, rs):
    print('Image:', imgFName)
    print('Ground Truth:', gtFName)
    print('Method:', methodName)
    print('Parameters: wsize =', wsize, ', thresCoef =', thresCoefOld)
    print('Radii:', rs)
# ----------------------------------------
def ReadGTInformation(gt_filename, gt_path, height):
    #gt_filename = gtFName
    gt_coords_name = os.path.join(gt_path, gt_filename + 'n.xlsx')
    gt_list_name = os.path.join(gt_path,gt_filename + '.csv')
    if os.path.isfile(gt_coords_name):
        gt_coords = pd.read_excel(gt_coords_name, header = None, nrows = 2, engine='openpyxl')  
    else:
        print("No ground-truth data available!")
    if os.path.isfile(gt_list_name):
        gt_unit = pd.read_csv(gt_list_name, sep = ';')
    else:
        print("No ground-truth unit available!")
    
    print("gt_coords_name:", gt_coords_name)

    length_pix = float(gt_unit[gt_unit['Unit']=='pixels']['Length'][0].replace(',', '.'))
    length_nm = np.linalg.norm(gt_coords[[0,1]].diff(),axis=1)[1]
    coeff_nm2pix = length_pix/length_nm

    gt_blobs = []
    for col in range(2,gt_coords.shape[1],2):
        i = col
        line = gt_coords[[i,i+1]]
        x,y = np.array(line.mean())*coeff_nm2pix
        r = np.linalg.norm(line.diff(),axis=1)[1]*coeff_nm2pix / 2.
        gt_blobs.append([height-y,x,r])
        
    gt_blobs = np.array(gt_blobs) 

    return gt_blobs

# -------------------------------------------
def MakeSheetOfParticles(img, blobs, pSize, nHor, gamma):
    # nHor = number particles on the sheet over X
    # gamma = true - gamma correction
    nBlobs = len(blobs)
    nVert = math.ceil(nBlobs / nHor)
    sheet = 255 * np.ones([nVert * (pSize + 1) - 1, nHor * (pSize + 1) - 1])
    i_blob = 0
    print(nBlobs, nHor, nVert)
    for j in range(nVert):
        for i in range(nHor):
            if i_blob < nBlobs:
                blob = blobs[i_blob]
                bx = int(round(blob[0]))
                by = int(round(blob[1]))
                minus = math.floor(pSize / 2)
                plus = math.ceil(pSize / 2)
                ib = img[bx - minus:bx + plus, by - minus:by + plus]  # subimage
                ib_x, ib_y = np.shape(ib)
                if ((ib_x == pSize) & (ib_y == pSize)):
                    # print(i_blob,':',j,i,':',j*(pSize+1),(j+1)*(pSize+1)-1,i*(pSize+1),(i+1)*(pSize+1)-1)
                    sheet[j * (pSize + 1):(j + 1) * (pSize + 1) - 1,
                    i * (pSize + 1):(i + 1) * (pSize + 1) - 1] = np.power(ib, gamma)
                else:
                    #print('blob ', i_blob, 'with center (', bx, ',', by, ') is out of range')
                    #print('bx-minus = ', bx-minus, 'bx+plus = ', bx+plus, 'by-minus = ', by-minus, 'by+plus = ', by+plus)
                    bxm = bx - minus
                    bxp = bx + plus
                    bym = by - minus
                    byp = by + plus
                    if (ib_x != pSize) & (ib_y == pSize):
                        if bxm < 0:
                            # print('sheet(bxm<0):', j * (pSize + 1) - bxm, (j + 1) * (pSize + 1) - 1, i * (pSize + 1), (i + 1) * (pSize + 1) - 1)
                            ib = img[bx - minus - bxm:bx + plus, by - minus:by + plus]
                            sheet[j * (pSize + 1) - bxm:(j + 1) * (pSize + 1) - 1,
                            i * (pSize + 1):(i + 1) * (pSize + 1) - 1] = np.power(ib, gamma)
                        if bxp<0:
                            #print('sheet(bxp<0):', j * (pSize + 1), (j + 1) * (pSize + 1) - 1 + bxp, i * (pSize + 1), (j + 1) * (pSize + 1) - 1 + bxp)
                            ib = img[bx - minus:bx + plus+bxp, by - minus:by + plus]
                            sheet[j * (pSize + 1) :(j + 1) * (pSize + 1) - 1 + bxp,
                            i * (pSize + 1):(i + 1) * (pSize + 1) - 1] = np.power(ib, gamma)
                    if (ib_x == pSize) & (ib_y != pSize):
                        if bym < 0:
                            # print('sheet(bym<0):', j * (pSize + 1), (j + 1) * (pSize + 1) - 1, i * (pSize + 1)-bym, (i + 1) * (pSize + 1) - 1)
                            ib = img[bx - minus-bxm:bx + plus, by - minus - bym:by + plus]
                            sheet[j * (pSize + 1)-bxm:(j + 1) * (pSize + 1) - 1,
                            i * (pSize + 1) - bym:(i + 1) * (pSize + 1) - 1] = np.power(ib, gamma)
                        if byp < 0:
                            # print('sheet(byp<0):', j * (pSize + 1), (j + 1) * (pSize + 1) - 1, i * (pSize + 1), (i + 1) * (pSize + 1) - 1+byp)
                            ib = img[bx - minus-bxm:bx + plus, by - minus:by + plus + byp]
                            sheet[j * (pSize + 1)-bxm:(j + 1) * (pSize + 1) - 1,
                            i * (pSize + 1):(i + 1) * (pSize + 1) - 1 + byp] = np.power(ib, gamma)
                    if (ib_x != pSize) & (ib_y != pSize):
                        if (bxm<0) & (bym<0):
                            # print('sheet(bxm<0):', j * (pSize + 1) - bxm, (j + 1) * (pSize + 1) - 1, i * (pSize + 1), (i + 1) * (pSize + 1) - 1)
                            ib = img[bx - minus - bxm:bx + plus, by - minus-bym:by + plus]
                            sheet[j * (pSize + 1) - bxm:(j + 1) * (pSize + 1) - 1,
                            i * (pSize + 1) - bym :(i + 1) * (pSize + 1) - 1] = np.power(ib, gamma)

                        if (bxm < 0) & (byp < 0):
                            ib = img[bx - minus-bxm:bx + plus, by - minus:by + plus + byp]
                            sheet[j * (pSize + 1)-bxm:(j + 1) * (pSize + 1) - 1,
                            i * (pSize + 1):(i + 1) * (pSize + 1) - 1 + byp] = np.power(ib, gamma)

                        if (bxp < 0) & (bym < 0):
                            #print('sheet(bxp<0):', j * (pSize + 1), (j + 1) * (pSize + 1) - 1 + bxp, i * (pSize + 1), (j + 1) * (pSize + 1) - 1 + bxp)
                            ib = img[bx - minus:bx + plus+bxp, by - minus - bym:by + plus]
                            sheet[j * (pSize + 1) :(j + 1) * (pSize + 1) - 1 + bxp,
                            i * (pSize + 1) - bym :(i + 1) * (pSize + 1) - 1] = np.power(ib, gamma)

                        if (bxp < 0) & (byp < 0):
                            #print('sheet(bxp<0):', j * (pSize + 1), (j + 1) * (pSize + 1) - 1 + bxp, i * (pSize + 1), (j + 1) * (pSize + 1) - 1 + bxp)
                            ib = img[bx - minus:bx + plus+bxp, by - minus:by + plus+byp]
                            sheet[j * (pSize + 1) :(j + 1) * (pSize + 1) - 1 + bxp,
                            i * (pSize + 1):(i + 1) * (pSize + 1) - 1+byp] = np.power(ib, gamma)


            i_blob = i_blob + 1
    return sheet
# ------------------------------------------------------------------
def ReadSheetOfNP(fName, wsize):
    img = cv2.imread(fName)

    fig, ax1 = plt.subplots(1, 1, figsize=(18,28), sharex=False, sharey=False)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    plt.tight_layout()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    szy, szx = np.shape(img)
    nVert = int((szy+1)/(wsize+1))
    nHor = int((szx+1)/(wsize+1))
    subimgs = []
    k = 0
    for j in range (nVert):
        for i in range (nHor):
            subimg = img[j*(wsize+1):j*(wsize+1)+wsize,i*(wsize+1):i*(wsize+1)+wsize]
            if sum(sum(subimg-255))!=0:
                subimgs.append(subimg)
                k = k+1
    return subimgs
# -----------------------------------------------------------------------------
def MakeFeatures(subimg, c1s, xy2,helpMatrs, mask):
    wsize, wsize = np.shape(subimg)
    imrow = np.reshape(subimg,(1,wsize*wsize))[0]
    f = np.zeros((16))
    f[0] = imrow[24]
    ind = np.array([17,23,24,25,31])
    f[1] = sum(imrow[ind])/len(ind)
    ind = np.array([16,17,18,23,24,25,30,31,32])
    f[2] = sum(imrow[ind])/len(ind)
    ind = np.array([16,17,18,23,24,25,30,31,32, 10, 22, 26, 38])
    f[3] = sum(imrow[ind])/len(ind)
    ind = np.array([0,1,7])
    f[4] = sum(imrow[ind])/len(ind)
    ind = np.array([5,6,13])
    f[5] = sum(imrow[ind])/len(ind)
    ind = np.array([35,42,43])
    f[6] = sum(imrow[ind])/len(ind)
    ind = np.array([41,47,48])
    f[7] = sum(imrow[ind])/len(ind)
    ind = np.array([0,1,7,5,6,13,35,42,43,41,47,48])
    f[8] = sum(imrow[ind])/len(ind)
    ind = np.array([0,1,2,4,5,6,7,8,12,13,14,20, 28,34,35,36,40,41,42,43,44,46,47,48])
    f[9] = sum(imrow[ind])/len(ind)
    f[10] = f[2]/f[9]
    ind = np.array([2,3,4,14,20,21,27,28,34,44,45,46])
    f[11] = sum(imrow[ind])/len(ind)
    ind = np.array([0,1,2,4,5,6,7,8,12,13,14,20, 28,34,35,36,40,41,42,43,44,46,47,48])
    f[12] = sum(imrow[ind])/len(ind)
    optc0,optc1, optcrit = mf.ApproxInWindowMask(wsize, subimg, c1s, xy2, helpMatrs, mask)
    f[13] = optc0
    f[14] = optc1
    f[15] = optcrit
    return f
# -------------------------------------------------------------------
def gtInformation(imgFName, img, gt_path):
    gt_base = os.path.splitext(os.path.split(imgFName)[1])[0]
    gt_blobs = []
    gtFName = gt_path + '/' + 'gt_blobs_' + gt_base + '.csv'

    with open(gtFName, encoding='utf-8') as gt_file:
        file_reader = csv.reader(gt_file, delimiter = ",")
        for row in file_reader:
            x = float(row[0])
            y = float(row[1])
            r = float(row[2])
            gt_blobs.append([x,y,r])
        gt_blobs = np.array(gt_blobs)

    if gt_base == '4-S1-no_area-100k-ordered':
        roi = [140, 533, 244, 336]
    else:
        roi = GetROI(img, gt_blobs)
    return gt_base, roi, gt_blobs
# ---------------------------------------------------------
def PrintAccuracy(match, fake, no_match):
    accuracy = match / (match+fake+no_match)
    print(f' Ground-truth particles detected by the algorithm: {match}\n',
          f' Ground-truth particles not detected by the algorithm: {no_match}\n',
          f' Non ground-truth particles detected: {fake}')
    accuracy = match / (match+fake+no_match)
    print(f'Accuracy: {accuracy}')
# ---------------------------------------------------------
def GenerateRandomCoordinates(n, shape, coords_occupied, pSize):
    # Generate n pairs of coordinates (y,x), that are not in coords_occupied
    szy, szx = shape
    hsz = pSize // 2
    k = 0
    for i in range(len(coords_occupied)):
        coords_occupied[i, 0] = int(round(coords_occupied[i, 0]))
        coords_occupied[i, 1] = int(round(coords_occupied[i, 1]))

    coords_rnd = np.zeros((n, 2))
    k = 0
    while k < n:
        rny = rnd.randint(hsz + 1, szy - hsz - 1)
        rnx = rnd.randint(hsz + 1, szx - hsz - 1)
        indy = np.where(np.array(coords_occupied[:, 0]) == rny)[0]
        if len(indy) == 0:
            coords_rnd[k] = [rny, rnx]
            k = k + 1
        else:
            blind = coords_occupied[indy]
            indx = np.where(np.array(blind[:, 1]) == rnx)[0]
            if len(indx) == 0:
                coords_rnd[k] = [rny, rnx]
                k = k + 1

    return coords_rnd
# --------------------------------------------------------------------------
def FindThresPrep(img, nbr=1000, thrPrepCoef=0.3):
    lm = peak_local_max(img, min_distance=3, threshold_abs=0, threshold_rel=None, footprint=None, labels=None)
    # print('nlm = ', np.shape(lm)[0])
    br = np.zeros(np.shape(lm)[0])
    for i in range(np.shape(lm)[0]):
        br[i] = img[lm[i, 0], lm[i, 1]]

    # print('minbr=', np.min(br[0:nbr]), 'maxbr=', np.max(br[0:nbr]), 'meanbr=', np.mean(br[0:nbr]))
    thres1 = np.mean(br[0:nbr])
    # print("thres1 = ", thres1, "thres1*", thrPrepCoef, "=", thres1*thrPrepCoef)
    thrPrep = thres1 * thrPrepCoef
    return thrPrep
# --------------------------------------------------------------------------
def AddBlobsToDoc(img, blobs, ws, nHor, gamma,fsz, _dpi, dwidth, title, doc):
    doc.add_paragraph(title)
    temp_fname = 'temp.png'
    sheet = MakeSheetOfParticles(img, blobs, ws, nHor, gamma)
    ysize, xsize = np.shape(sheet)
    sheet_numbers = 255 * np.ones((ws, xsize))
    for i in range (4, nHor, 5):
        cv2.putText(sheet_numbers, str(i+1), ((ws+1)*i,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, fsz, 0, 1)

    sheet = np.row_stack((sheet_numbers, sheet))
    fig, ax = plt.subplots(1, 1, figsize=(18,28), sharex=False, sharey=False)
    ax.imshow(sheet, cmap='gray')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(temp_fname, dpi = _dpi, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,transparent=False, bbox_inches='tight', pad_inches=0)
    doc.add_picture(temp_fname, width = docx.shared.Cm(dwidth))

# ] ----- fuctions ------------------------


if __name__ == '__main__':
    
	(script,
	nproc,		# number of processes
	ntimes,	    # number of runs
	imgFName,	# File name for processing
	wsize,		# window size for approximation
	rs_start,	# start value of possible nanoparticle radius
	rs_end,	    # end value of possible nanoparticle radius
	rs_step,	# step for changing nanoparticle radius
	tCO_start,
	tCO_end,
	tCO_step,
	res_path,
	gt,             #True/False - Ground truth information availability
	gt_path) = argv

	nproc = int(nproc)
	ntimes = int(ntimes)	
	wsize = int(wsize)
	rs_start = float(rs_start)
	rs_end = float(rs_end)
	rs_step = float(rs_step)

	tCO_start = float(tCO_start)
	tCO_end = float(tCO_end)
	tCO_step = float(tCO_step)
	
	print('script', script)
	print('nproc:', nproc)
	print('ntimes:', ntimes)
	print('imgFName:', imgFName)
	print('wsize:', wsize)
	print('rs: ', rs_start, rs_end, rs_step)
	print('tCO: ', tCO_start, tCO_end, tCO_step)
	print('res_path:', res_path)
	print('gt:', gt)
	print('gt_path', gt_path)

	FIGSIZE = (9,7)
	ORDERED_DSET_PATH = 'Data'
	GROUND_TRUTH_PATH = 'Ground_truth'
	MATLAB_EXP_PATH = 'Matlab_data'

	methodName = "Exponential Approximation"
	print("!!! РїСЂРµС„РёР»СЊС‚СЂ med+th, РѕР±СЂР°Р±РѕС‚РєР° th !!!")

	# РІРѕР·РјРѕР¶РЅС‹Рµ СЂР°РґРёСѓСЃС‹ РЅР°РЅРѕС‡Р°СЃС‚РёС†
	rs = np.arange(float(rs_start), float(rs_end), float(rs_step))
	c1s = 1/(rs*rs)

	img, height, img_init = Preprocessing(imgFName, False, True)   # С‡С‚РµРЅРёРµ Рё РїСЂРµРґРѕР±СЂР°Р±РѕС‚РєР° (topHat)
	img_med_th, height, img_init = Preprocessing(imgFName, True, True)   # С‡С‚РµРЅРёРµ Рё РїСЂРµРґРѕР±СЂР°Р±РѕС‚РєР° (median+topHat)

	points = img_med_th>10#np.mean(img_med_th)  # РєРѕРѕСЂРґРёРЅР°С‚С‹ С†РµС‚СЂРѕРІ РѕРєРѕРЅ РґР»СЏ Р°РїРїСЂРѕРєСЃРёРјР°С†РёРё (Р±СѓРґРµРј СЂР°СЃСЃРјР°С‚СЂРёРІР°С‚СЊ С‚РѕР»СЊРєРѕ РѕРєРЅР°, СЏСЂРєРѕСЃС‚СЊ РєРѕС‚РѕСЂС‹С… РІ С†РµРЅС‚СЂРµ > 10)                            
	wsize = 7  # СЂР°Р·РјРµСЂ РѕРєРЅР° Р°РїРїСЂРѕРєСЃРёРјР°С†РёРё
	thresCoefNew = 0.25 # РјР°СЃС€С‚Р°Р±РёСЂСѓСЋС‰РёР№ РєРѕСЌС„С„РёС†РёРµРЅС‚ РґР»СЏ РїРѕСЂРѕРіР° (РЅРѕРІС‹Р№ РІР°СЂРёР°РЅС‚ РІС‹С‡РёСЃР»РµРЅРёСЏ)
	thresCoefOlds = np.arange(tCO_start, tCO_end, tCO_step) # РјР°СЃС€С‚Р°Р±РёСЂСѓСЋС‰РёР№ РєРѕСЌС„С„РёС†РёРµРЅС‚ РґР»СЏ РїРѕСЂРѕРіР° (СЃС‚Р°СЂС‹Р№ РІР°СЂРёР°РЅС‚ РІС‹С‡РёСЃР»РµРЅРёСЏ)
	#thresCoefOlds = np.arange(0.35, 0.5, 0.05) # РјР°СЃС€С‚Р°Р±РёСЂСѓСЋС‰РёР№ РєРѕСЌС„С„РёС†РёРµРЅС‚ РґР»СЏ РїРѕСЂРѕРіР° (СЃС‚Р°СЂС‹Р№ РІР°СЂРёР°РЅС‚ РІС‹С‡РёСЃР»РµРЅРёСЏ)
	thresType = "old"  # РІР°СЂРёР°РЅС‚ РІС‹С‡РёСЃР»РµРЅРёСЏ РїРѕРїСЂРѕРіР° (new/old)
	faster = False  # РЎСѓС‰РµСЃС‚РІРµРЅРЅРѕ Р±С‹СЃС‚СЂРµРµ, РЅРѕ С‡СѓС‚СЊ РјРµРЅРµРµ С‚РѕС‡РЅРѕ (РІС‹РєРёРґС‹РІР°РµРј РёР· СЂР°СЃСЃРјРѕС‚СЂРµРЅРёСЏ РѕРєРЅР°, СЃСЂРµРґРЅСЏСЏ СЏСЂРєРѕСЃС‚СЊ РєРѕС‚РѕСЂС‹С… РјРµРЅСЊС€Рµ СЃСЂРµРґРЅРµР№ СЏСЂРєРѕСЃС‚Рё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ)

	#gt = True # РµСЃС‚СЊ ground truth РёРЅС„РѕСЂРјР°С†РёСЏ


	vis = False #!!!!
	#vis = True # РІРёР·СѓР°Р»РёР·Р°С†РёСЏ 


	print("max =", np.max(img))
	print("mean =", np.mean(img))
	print("median =", np.median(img))

	if gt: 
		# Р Р°Р·РјРµС‚РєР° СЌРєСЃРїРµСЂС‚Р°
		
		gt_base = os.path.splitext(os.path.split(imgFName)[1])[0]
		gt_blobs = []
		gtFName = gt_path + '/' + 'gt_blobs_' + gt_base + '.csv'

		with open(gtFName, encoding='utf-8') as gt_file:
			file_reader = csv.reader(gt_file, delimiter = ",")
			for row in file_reader:
				x = float(row[0])
				y = float(row[1])
				r = float(row[2])
				gt_blobs.append([x,y,r])
			gt_blobs = np.array(gt_blobs)

		if gt_base == '4-S1-no_area-100k-ordered':
			roi = [140, 533, 244, 336]		
		else:
			roi = GetROI(img)

	else:
		roi = [0, 0, np.shape(img)[0], np.shape(img)[1]]
		data2show = {'results':1, 'ground truth':0, 'missed gt':0, 'fake est':0, 'large':1}
	
	res = np.zeros((np.shape(thresCoefOlds)[0], 5))
	for i in range (np.shape(thresCoefOlds)[0]):
		thresCoefOld = thresCoefOlds[i]
		PrintHeader(imgFName, gtFName, methodName, thresType, thresCoefOld, thresCoefNew, faster, wsize, rs)
		times = np.zeros((ntimes,1))
		for tt in range(ntimes):
			before = time.time()
			blobs_est = ExponetialApproximation(img, c1s, points, faster, wsize, thresType, thresCoefNew, thresCoefOld, nproc)
			after = time.time()
			exp_time = after - before
			#print('tt = ', tt, 'exp_time:', exp_time)
			times[tt] = exp_time
		print('times:', times)

		accuracy = 0
		if gt:
			match, no_match, fake, no_match_gt_blobs, fake_blobs = accur_estimation2(gt_blobs, blobs_est, roi=roi, thres=0.25)
			accuracy = match / (match+fake+no_match)
			print(f' Ground-truth particles detected by the algorithm: {match}\n',
				f'Ground-truth particles not detected by the algorithm: {no_match}\n',
				f'Non ground-truth particles detected: {fake}')
			accuracy = match / (match+fake+no_match)
			print(f'Accuracy: {accuracy}')
       
		print("\n *** thresCoefOld = ", thresCoefOld, ", acc = ", accuracy)
		res[i][0] = thresCoefOld
		res[i][1] = accuracy
		res[i][2] = np.mean(times)
		res[i][3] = np.std(times)
		res[i][4] = ntimes


    ##    # РІС‹С‡РёСЃР»РµРЅРёРµ РѕС‚РєР»РѕРЅРµРЅРёР№ СЂР°РґРёСѓСЃРѕРІ РЅР°Р№РґРµРЅРЅС‹С… С‡Р°СЃС‚РёС† РѕС‚ СЂР°Р·РјРµС‡РµРЅРЅС‹С… (РЅР°Р№РґРµРЅРЅС‹Р№ РјРёРЅСѓСЃ СЂР°Р·РјРµС‡РµРЅРЅС‹Р№)
    ##    rdif_mean, rdif_std, rdif, rgt, rest = accur_radii(gt_blobs, blobs_est, roi, thres=0.25)
    ##    print("mean radius difference = ", rdif_mean, ", std = ", rdif_std)
    ##    print("Histograms for ground truth radii (left), computed radii (center) and difference (right)")
    ##    size = 3
    ##    fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(size*3,size), sharex=False, sharey=False)
    ##    ax_1.hist(rgt, bins='auto') 
    ##    ax_2.hist(rest, bins='auto') 
    ##    ax_3.hist(rdif, bins='auto')
    ##    plt.show()
        
	data2show = {'results':1, 'ground truth':1, 'missed gt':1, 'fake est':1, 'large':1}
       
	if vis:
		data2show = {'results':1, 'ground truth':1, 'missed gt':1, 'fake est':1, 'large':1}
		#res_path = 'e:\From_external\WorkNew\PYTHON\Microscopy\Ground_truth\\res1'
		#res_path = 'c:\Valentina\PYTHON\Mycroscopy\Ground_truth\\res'
		imFName = os.path.splitext(os.path.split(imgFName)[1])[0]
		#print('rs shape[0]:', np.shape(rs)[0])
		mode = '_pref-medth_obr-th'
		rstr = '_r'+ str(round(rs[0],2))+ '-' + str(round(rs[np.shape(rs)[0]-1],2))
		thresstr = '_thCO' + str(round(thresCoefOld,2))
		accstr = '_acc'+ str(round(accuracy,4))
		outfname = os.path.join(res_path, imFName + mode + rstr  + thresstr + accstr+'_init-roi.png')
		print(outfname)
		#def Visualization(temp_img, data2show, roi, show=True, save = False, outfname='', _dpi=300, lsize=100):
		Visualization(img_init, data2show, roi, _show=False, _save=True, outfname=outfname, _dpi=147.2, lsize=5, lwidth=1)
		outfname = os.path.join(res_path, imFName + mode + rstr + thresstr + accstr+'_prep-roi.png')
		print(outfname)

		Visualization(img, data2show, roi, _show=False, _save=True, outfname=outfname, _dpi=147.2, lsize=5, lwidth=1)
            
		roi1 = [0, 0, np.shape(img)[0], np.shape(img)[1]]
		outfname = os.path.join(res_path, imFName + mode + rstr + thresstr + accstr + '_init-full.png')
		print(outfname)
		Visualization(img_init, data2show, roi1, _show=False, _save=True, outfname=outfname, _dpi=1000, lsize=5, lwidth=0.2)
           
		outfname = os.path.join(res_path, imFName + mode + rstr + thresstr + accstr + '_prep-full.png')
		print(outfname)
		Visualization(img, data2show, roi1, _show=False, _save=True, outfname=outfname, _dpi=1000, lsize=5, lwidth=0.2)
        
	print(res)    
