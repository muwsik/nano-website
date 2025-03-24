# -*- coding: cp1251 -*-

import numpy as np
import cv2
import csv
from skimage import morphology
import matplotlib.pyplot as plt
import os
import math
import time
import sys
import multiprocessing
from skimage.feature import peak_local_max


# !!!!!!!!!!! 
def ApproxInWindow_NormExp(wsize, z, c1s, xy2, helpMatrs, npar):
    """Approximation of function z by the exponential function
    c0*exp(-c1(x^2+y^2))+c2
    npar = 2 - определяются только параметры c0 и с1 (с2=0)
    npar = 3 - определяются все параметры
    c1s - vector of c1 values, for each of which the approximation is made (only c0 is determined)
    then the optimal c1 is found
    z is a matrix [wsize x wsize]
    ! NO MASK! only full window is used! 
    helpMatrs is a spacial structure with precomputed matrixes and SLAE coefficients to found с0
    !!! it is supposed that helpMatrs shape and z shape are matched!!!
    оптимальное c1 выбирается по минимуму критерия МНК,
    а возвращается и значение критерия, и нормированная ошибка!!!!
    """
    if (z == 0).all():
        return 0, 0, 0

    hsz = wsize // 2
    nc = np.shape(c1s)
    norm_errors = np.zeros(nc, dtype='float')
    
    critsMNK = np.zeros(nc, dtype='float')
    c0s = np.zeros(nc, dtype='float')
    c2s = np.zeros(nc, dtype='float')
    zapr = np.zeros((wsize, wsize), dtype='float')
    npoints = wsize*wsize
    i = 0
    for c1 in c1s:
        # determine the optimal c0 и с2 for the given c1
        m = helpMatrs.get(c1)
        a11 = sum(sum(np.array(m) ** 2))
        b1 = sum(sum(z * m))  
        match npar:
            case 2:
                c0 = b1 / a11
                c2 = 0
            case 3:
                b2 = sum(sum(z)) 
                a22 = npoints
                a12 = sum(sum(m))
                c2 = (b2 - b1)/(a22 - a12)
                c0 = (b1 - c2*a12)/a11
        
        zapr = c0 * np.exp(-c1 * xy2) + c2        
        delta = z - zapr 
        
        chisl = np.sqrt( sum( sum(np.array(delta) * delta) )/npoints)
        znam1 = np.sqrt( (sum( sum(np.array(z, dtype='float64')*z) ))/npoints)
        znam2 = np.sqrt( (sum( sum(np.array(zapr)*zapr) ))/npoints)        
        norm_errors[i] = chisl / ( znam1 +znam2 ) 
        critsMNK[i] = np.sqrt(sum( sum(np.array(delta) * delta))/npoints)            

        #print('npoints=', npoints, ' znam1=',znam1)
        c0s[i] = c0
        c2s[i] = c2
        i = i + 1

    indMNK = np.argmin(critsMNK)  
    indErr = np.argmin(norm_errors)
    radMNK = 1 / np.sqrt(c1s[indMNK])
    radErr = 1 / np.sqrt(c1s[indErr])
    #if (indMNK!=indErr):
        #print("iMNK=%d iE=%d rMNK=%.1f rE=%.1f crMNK[MNK]=%.3f crMNK[E]=%.3f er[MNK]=%.3f er[E]=%.3f" % 
        #      (indMNK,  indErr,    radMNK,   radErr,   critsMNK[indMNK],   critsMNK[indErr], norm_errors[indMNK],  norm_errors[indErr]))
        
    optc0 = c0s[indMNK]
    optc1 = c1s[indMNK]
    optc2 = c2s[indMNK]
    opt_norm_error = norm_errors[indMNK] # оптимальное c1 выбирается по минимуму критерия МНК,
                         # а возвращается нормированное значение критерия (показатель качества)
    
    return optc0, optc1, optc2, critsMNK[indMNK], opt_norm_error


def MakeHelpMatricesNew(wsize, rs):
    """computation of help matrixes to solve SLAE in MNK for approximation
    by the function c0*exp(-c1(x^2+y^2)) with discrete values of c1
    wsize - size of approximation window
    rs - vector of possible radii
    """
    cs = cs = 1 / (rs * rs)  # возможные значения c1
    hsz = wsize // 2  # max and min values of positive and negative coordinate values
    xx = np.array((np.arange(-hsz, hsz+1)))
    xx = xx*xx
    xy2 = np.array([xx]*wsize).T + np.array([xx]*wsize)

    helpMatrs = {}
    for c1 in cs:
        m = np.exp(-c1 * xy2)
        helpMatrs.update({c1: m})
    return helpMatrs, xy2

def ImageReading(imgFName):
    image = cv2.imread(imgFName)
    height = image.shape[0]
    image = image[:890,:]
    #print(np.shape(image))
    img_init = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#image[:,:,2] # Blue conponent is used    
    #print(np.shape(img_init))
    return img_init, height   

def PreprocessingMedian(img, med_sz = 3):
    from skimage.filters import median
    #footprint =  np.ones((med_sz, med_sz))
    footprint =  np.ones((med_sz, med_sz))
    img2bin = median(img, footprint=footprint)
    return img2bin

def PreprocessingTopHat(img, th_sz = 4):
    selem =  morphology.disk(th_sz)
    img2bin = morphology.white_tophat(img, selem)
    return img2bin

def GetSubimage(img, point, wsize): 
    '''
    Извлечение из img окошка с центром в point и размером wsize
    Если окно не помещается целиком (точка близко к краю), то возвращается False и пустая матрица
    '''
    hfwsz = int(wsize/2)
    x = int(point[0])
    y = int(point[1])
    if (x-hfwsz>=0) and (x+hfwsz+1<np.shape(img)[0]) and (y-hfwsz>=0) and (y+hfwsz+1<np.shape(img)[1]):
        subimg = img[x-hfwsz:x+hfwsz+1, y-hfwsz:y+hfwsz+1]
        success = True
    else: 
        subimg = 0
        success = False
    return success, subimg


def VisualizationSimple(temp_img, data2show, blobs_est, roi, FIGSIZE = (9,7), _show=True, _save = False, outfname='', _dpi=300, lsize=100, lwidth = 1, nums = True):
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
                #print(i,": (", int(round(y)), ",",int(round(x)),", r=",r)        
                if data2show['large'] and r>=lsize:
                    _color = color[4]
                else: 
                    _color = color[1]
                c = plt.Circle((x, y), r, color=_color, linewidth=lwidth, fill=False)
                ax.add_patch(c)
                
            if nums == True:
                plt.text(x+3, y+3, str(i-1), fontsize=7, color = 'white')
           
    ax.set_axis_off()

    plt.tight_layout()
    if _show:
        plt.show()
    if _save:
        plt.savefig(outfname, dpi=_dpi, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,transparent=False, bbox_inches='tight', pad_inches=0)     

# ------------------------------------------------------
def VisualizationSimpleNums(temp_img, blobs_est, FIGSIZE = (9,7), _show=True, _save = False, outfname='', _dpi=300, lsize=100, lwidth = 1, nums = True):
    
    # show = True - to show on a display
    # save = True - to save in a file outfname
    # _dpi - resolution 
    # lwidth - line width
    
    color = ['blue', 'lime', 'red', 'yellow', 'magenta'] #ground truth, calculated, not found, fake, large
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True, sharey=True)

    ax.imshow(temp_img, cmap='gray')

    i = 0
    for blob in blobs_est:
        x = blob[1]
        y = blob[0]
        r = blob[2]
        _color = color[1]
        c = plt.Circle((x, y), r, color=_color, linewidth=lwidth, fill=False)
        ax.add_patch(c)
        if nums == True:
            plt.text(x+3, y+3, str(i), fontsize=7, color = 'white')
        i = i+1
           
    ax.set_axis_off()

    plt.tight_layout()
    if _show:
        plt.show()
    if _save:
        plt.savefig(outfname, dpi=_dpi, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,transparent=False, bbox_inches='tight', pad_inches=0)     
# -----------------------------------------------------------------------
        

def ExtractPointsBySquareMask(point, imgshape, msk):
    points = np.zeros(imgshape)        
    pntmask = np.ones((msk,msk))
    hfmsk = int(msk/2)
    x = int(point[0])
    y = int(point[1])
    #points[lm[i][0]-hfmsk:lm[i][0]+hfmsk+1, lm[i][1]-hfmsk:lm[i][1]+hfmsk+1] = pntmask
    points[x-hfmsk:x+hfmsk+1, y-hfmsk:y+hfmsk+1] = pntmask
    ppx,ppy = np.where(points==1)
    return ppx, ppy

def ApproximationWithFindingTheBestCenter(img, point, xy2, helpMatrs, params, prn):
    # рассматриваем точку локального максимума и точки вокруг него 
    # из окошка размером params["msk"] 
    # для уточнения координат центра наночастицы, который не всегда совпадает
    # с локальным максимумом
    # всего npnts=msk*msk точек,  ppx, ppy - список координат этих точек
    # пороги для отбора точек, удовлетворяющих условию: 
    # params["thr_c0"], params["thr_c1"], params["thr_error"]
    # params["best_mode"] = 1 или 2: среди точек, удовлетворяющих условиям, выбираем наилучшую 
    # по c1 (best_mode = 1)  или по нормированной ошибке (best_mode <> 1)
    # params["met"] = 'exp'
    # params["npar"] = 3 # число параметров аппроксимации 3 или 2
    
    msk = params["msk"]
    wsize = params["wsize"]
    rs = params["rs"]
    c1s = 1/(rs*rs)
    thr_c0 = params["thr_c0"]
    thr_c1 = params["thr_c1"]
    thr_c2 = params["thr_c2"]
    thr_error = params["thr_error"] 
    best_mode = params["best_mode"]
    npar = params["npar"]
    met = params["met"]
    #print('point=', point)
    ppx, ppy = ExtractPointsBySquareMask(point, np.shape(img), msk)
    npoints =np.shape(ppx)[0]  
    opt_error = 100
    optc0 = 0
    optc1 = 0
    optc2 = 0
    blob_best = [0.,0.,0.]
    best_found = False
    ibest = 0
    for pp in range(npoints):
        # вырезаем окошко subimg из изображения img
        success, subimg = GetSubimage(img, (ppx[pp],ppy[pp]), wsize)
        if success:
            # непосредственно аппроксимация
            if met=="exp":
                c0, c1, c2, crit, norm_error = ApproxInWindow_NormExp(wsize, subimg, c1s, xy2, helpMatrs, npar)        
            if ((c0 > thr_c0) and (norm_error<=thr_error) and (c1 > thr_c1) and (c2 <= thr_c2)): 
                if(prn):
                    print("%d (%d, %d): r=%.1f, c0=%.3f, c1=%.3f, c2=%.3f, error=%.4f TRUE" % (pp, ppy[pp], ppx[pp], r, c0, c1, c2, norm_error))
                best_found = True
                is_better = False
                if best_mode ==1: 
                    if c1>optc1: 
                        is_better = True
                else:
                    if norm_error<opt_error:
                        is_better = True
                if is_better:        
                    ibest = pp
                    opt_error = norm_error
                    optc0 = c0
                    optc1 = c1
                    optc2 = c2
                    rad = 1 / np.sqrt(c1)
                    blob_best = np.array([ppx[ibest], ppy[ibest], rad])  #!!  blob_best = (ppx[ibest], ppy[ibest], rad)
            else:
                if (prn):
                    print("%d (%d, %d): r=%.1f, c0=%.3f, c1=%.3f, c2=%.3f, error=%.4f FALSE" % (pp, ppy[pp], ppx[pp], r, c0, c1, c2, norm_error))
    return best_found, blob_best, optc0, optc1, optc2, opt_error               

def ApproximationWithFindingTheBestCenter_NoFiltering(img, point, xy2, helpMatrs, params, prn):
    # рассматриваем точку локального максимума и точки вокруг него 
    # из окошка размером params["msk"] 
    # для уточнения координат центра наночастицы, который не всегда совпадает
    # с локальным максимумом
    # всего npnts=msk*msk точек,  ppx, ppy - список координат этих точек
    # params["best_mode"] = 1 или 2: среди точек, удовлетворяющих условиям, выбираем наилучшую 
    # по c0 (best_mode = 1), по с1 (best_mode = 2) или по нормированной ошибке (иначе)
    # params["met"] = 'exp'
    # params["npar"] = 3 # число параметров аппроксимации 3 или 2
    
    msk = params["msk"]
    wsize = params["wsize"]
    rs = params["rs"]
    c1s = 1/(rs*rs)
    best_mode = params["best_mode"]
    npar = params["npar"]
    met = params["met"]
    
    ppx, ppy = ExtractPointsBySquareMask(point, np.shape(img), msk)
    npoints =np.shape(ppx)[0]  
    opt_error = 100
    optc0 = 0
    optc1 = 0
    optc2 = 0
    blob_best = [0.,0.,0.]
    ibest = 0
    for pp in range(npoints):
        # вырезаем окошко subimg из изображения img
        success, subimg = GetSubimage(img, (ppx[pp],ppy[pp]), wsize)
        if success:
            # непосредственно аппроксимация
            if met=="exp":
                c0, c1, c2, crit, norm_error = ApproxInWindow_NormExp(wsize, subimg, c1s, xy2, helpMatrs, npar)        
            if(prn):
                print("     %d (%d, %d): r=%.1f, c0=%.3f, c1=%.3f, c2=%.3f, error=%.4f " % (pp, ppy[pp], ppx[pp], 1 / np.sqrt(c1), c0, c1, c2, norm_error))
            is_better = False
            match best_mode:
                case 1:
                    if c1>optc1: 
                        is_better = True
                case 2:
                    if c0>optc0:
                        is_better = True
                case _:
                    if norm_error<opt_error:
                        is_better = True
            if is_better:        
                ibest = pp
                opt_error = norm_error
                optc0 = c0
                optc1 = c1
                optc2 = c2
                rad = 1 / np.sqrt(c1)
                blob_best = np.array([ppx[ibest], ppy[ibest], rad])  #!!  blob_best = (ppx[ibest], ppy[ibest], rad)
            
    return blob_best, optc0, optc1, optc2, opt_error               


def FilterBlobs(blobs_ext, params):
    thr_c0 = params["thr_c0"]
    thr_r = params["thr_r"]
    thr_c2 = params["thr_c2"]
    thr_error = params["thr_error"]

    filtered_blobs = []
    blobs_rest = []
    for blob in blobs_ext:
        r = blob[2]
        c0 = blob[3]
        #c1 = blob[4]
        c2 = blob[5]
        norm_error = blob[6]
        #if ((c0 > thr_c0) and (norm_error<=thr_error) and (c1 > thr_c1) and (c2 <= thr_c2)):
        if ((c0 > thr_c0) and (norm_error<=thr_error) and (r <= thr_r) and (c2 <= thr_c2)):
            filtered_blobs.append(blob)
        else:
            blobs_rest.append(blob)
    return filtered_blobs, blobs_rest
 
#if ((c0 > thr_c0) and (norm_error<=thr_error) and (c1 > thr_c1) and (c2 <= thr_c2)):

def PrefilteringPoints(img, min_dist, thrBr):
    # поиск локальных максимумов и отбрасывание тех, которые по яркости меньше thrBr
    # минимальное расстояние между локальными максимумами min_dist
    lm = peak_local_max(img, min_distance=min_dist, threshold_abs=0, threshold_rel=None, footprint=None, labels=None)
    nlm = np.shape(lm)[0]
    print('точек до фильтрации:', nlm)
    for i in range(nlm-1,-1,-1):
        if img[lm[i,0], lm[i,1]]<thrBr:    
            lm = np.delete(lm,[i],0)
    nlmax = np.shape(lm)[0]
    print('точек после фильтрации:', nlmax)
    return lm, nlmax

def PaintBlobProfiles(img_init,img_prep, blob, wsize, dop, c0,c1,c2, met):
    x = int(blob[0])
    y = int(blob[1]) 
    hfwsz = int(wsize/2) + dop
    ptHor = img[x-hfwsz:x+hfwsz+1, y]
    ptVert = img[x, y-hfwsz:y+hfwsz+1]
    #print(img[x-hfwsz:x+hfwsz+1, y-hfwsz:y+hfwsz+1])
    xx =np.array(range(-hfwsz,hfwsz+1,1))
    xx2 =np.arange(-hfwsz,hfwsz+1, 0.1)
    success, subimg_init = GetSubimage(img_init, blob, wsize+dop)
    success, subimg_prep = GetSubimage(img_prep, blob, wsize+dop)
    appr = c0*np.exp(-c1*xx2*xx2)+c2
    zapr = c0 * np.exp(-c1 * xy2) + c2  
    #print('appr shape: ', np.shape(zapr))
    #print('subimg=', subimg_prep)
    #print('zapr=', zapr)
    #print('delta=', subimg_prep-zapr)
    _min = np.min(np.min(zapr))
    _max = np.max(np.max(zapr))
    fig, axs = plt.subplots(1, 4, figsize=[8,3], sharex=True, sharey=True)
    a0 = axs[0].imshow(subimg_init, norm = None, cmap = 'magma') #cmap='gray')
    a1 = axs[1].imshow(subimg_prep, norm = None, cmap = 'magma') #, cmap='gray')
    a2 =axs[2].imshow(zapr, norm = None, cmap = 'magma') #, cmap='gray')
    a3 =axs[3].imshow(np.abs(subimg_prep-zapr), norm = None, vmin = _min, vmax = _max, cmap = 'magma') # cmap='gray')
    fig.colorbar(a0, ax=axs[0])
    fig.colorbar(a1, ax=axs[1])
    fig.colorbar(a2, ax=axs[2])
    fig.colorbar(a3, ax=axs[3])
    
    
    plt.show()  
    fig, ax = plt.subplots(1, 2, figsize=[3,2], sharex=True, sharey=True)
    ax[0].plot(xx, ptHor, 'ob', xx2, appr,'-r')
    ax[1].plot(xx, ptVert,'ob', xx2, appr,'-r')
    plt.show()  
  
            
# -------------------------------------------------------------------
def BlobsExp_Info(blobs_extended, paint_profiles, params):
    # вывод информации о частицах, найденных при помощи экспоненциальной аппроксимации
    t = 0
    for blob in blobs_extended:
        y = int(round(blob[0]))
        x = int(round(blob[1]))
        r = blob[2]
        c0 = blob[3]
        c1 = blob[4]
        c2 = blob[5]
        norm_error = blob[6]
        print("%d (%d, %d): r=%.1f, c0=%.3f, c1=%.3f, c2=%.3f, error=%.4f" % (t, int(round(y)), int(round(x)), r, c0, c1, c2, norm_error))
        if paint_profiles:
            PaintBlobProfiles(img_init, img_med_th, blob, params["wsize"], 0, c0,c1,c2, params["met"])
        t = t+1

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
# -----------------------------------------------------------
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
# --------------------------------------------------------
def PrintAccuracy(match, fake, no_match):
    accuracy = match / (match+fake+no_match)
    print(f' Ground-truth particles detected by the algorithm: {match}\n',
          f' Ground-truth particles not detected by the algorithm: {no_match}\n',
          f' Non ground-truth particles detected: {fake}')
    accuracy = match / (match+fake+no_match)
    print(f'Accuracy: {accuracy}')
# -------------------------------------------------

#def Visualization(temp_img, data2show, roi, blobs_est, gt_blobs, fake_blobs, no_match_gt_blobs):
#def Visualization_GT(temp_img, data2show, blobs_est, gt_blobs, fake_blobs, no_match_gt_blobs, roi, FIGSIZE = (9,7), _show=True, _save = False, outfname='', _dpi=300, lsize=100, lwidth = 1, nums=True):
def Visualization_GT(temp_img, data2show, match_blobs, truedetected_blobs, fake_blobs, no_match_gt_blobs, roi, FIGSIZE = (9,7), _show=False, _save = True, outfname='', _dpi=300, lsize=100, lwidth = 1, nums=True):

    #data2show = {'match gt':1, 'true detected':1, 'missed gt':1, 'fake est':1, 'large':1}
    # large - to colrize particles, that are lager then lsize
    # show = True - to show on a display
    # save = True - to save in a file outfname
    # _dpi - resolution 
    # lwidth - line width
    # nums - to print blobs numbers
    
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

    if data2show['match gt']:
        #print("\n estimated:")
        i = 0
        for blob in match_blobs:
            y, x, r = blob
               
            if x>=roi[1] and x<=roi[1]+roi[3] and y>=roi[0] and y<=roi[0]+roi[2]:
                i = i+1
                #print(i,": (", int(round(y)), ",",int(round(x)),";",r,": c0=",optc0s7[int(round(y)),int(round(x))], " crit=",optcrits7[int(round(y)),int(round(x))])        
                c = plt.Circle((x, y), r, color= color[0], linewidth=lwidth, fill=False)
                ax.add_patch(c)
                if nums == True:
                    plt.text(x-7, y-7, str(i-1), fontsize=7, color = color[0])

    if data2show['true detected']:         
        i = 0
        for blob in truedetected_blobs:
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
                if nums == True:
                    plt.text(x+3, y+3, str(i-1), fontsize=7, color = color[1])
   
    if data2show['missed gt']:
        #print("\n gt:")
        i = 0
        for blob in no_match_gt_blobs:
            y, x, r = blob
            i = i+1
            # print(i, ": (", int(round(y)), ",",int(round(x)),";",r,": c0=",optc0s7[int(round(y)),int(round(x))], " crit=",optcrits7[int(round(y)),int(round(x))])
            c = plt.Circle((x, y), r, color=color[2], linewidth=lwidth, fill=False)
            ax.add_patch(c)
            if nums == True:
                plt.text(x+3, y+3, str(i-1), fontsize=7, color = color[2])

    if data2show['fake est']:
        #print("\n missed gt:")
        i = 0
        for blob in fake_blobs:
            y, x, r = blob
            i = i+1
            xx = int(round(x))
            yy = int(round(y))

            #print(i, ": (", yy, ",",xx,";",r,": c0=",optc0s7[yy,xx], " crit=",optcrits7[yy,xx])
            #print("(", yy-roi[0], ", ", xx-roi[1], ")")
        
            #print(yy-2, yy+2, xx-2, xx+2)
            #print(optc0s7[yy-2:yy+3, xx-2:xx+3])
            #print(optcrits7[yy-2:yy+3, xx-2:xx+3])
            c = plt.Circle((x, y), r, color=color[3], linewidth=lwidth, fill=False)
            ax.add_patch(c)   
            if nums == True:
                plt.text(x+3, y+3, str(i-1), fontsize=7, color = color[3]) 
        
    ax.set_axis_off()

    plt.tight_layout()
    if _show:
        plt.show()
    if _save:
        plt.savefig(outfname, dpi=_dpi, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,transparent=False, bbox_inches='tight', pad_inches=0)                    
# ---------------- END LIB ---------------------




# ---------------- CACHE FUN ---------------------
import streamlit as st

@st.cache_data(show_spinner = False)
def CACHE_HelpMatricesNew(_wsize, _rs):
    return MakeHelpMatricesNew(_wsize, _rs)

#@st.cache_data(show_spinner = False)
def CACHE_PrefilteringPoints(_img, _sz_med, _sz_th, _min_dist, _thr_br):
    img_med = PreprocessingMedian(_img, _sz_med)
    img_med_th = PreprocessingTopHat(img_med, _sz_th) 
    
    lm, nlmax = PrefilteringPoints(img_med_th, _min_dist, _thr_br)

    return lm, img_med_th

#@st.cache_data(show_spinner = "Nanoparticle detection on process...")
def CACHE_ExponentialApproximationMask_v3(_img, _lm, _xy2, _helpMatrs, _params, _prn = False):
    number_blobs = len(_lm)

    blobs_full = np.zeros([number_blobs, 3])      # blobs_full[i] = y, x, r
    values_full = np.zeros([number_blobs, 4])     # values_full[i] = c0, c1, c2, norm_error

    for i, temp_lm in enumerate(_lm):
        blob, c0, c1, c2, norm_error = ApproximationWithFindingTheBestCenter_NoFiltering(
            _img,    
            temp_lm,
            _xy2,
            _helpMatrs,
            _params,
            _prn
        )

        blobs_full[i, :] = blob
        values_full[i, :] = c0, c1, c2, norm_error

    return blobs_full, values_full

def my_FilterBlobs_change(blobs_ext, blobs_params, params):
    thr_c0 = params["thr_c0"]
    thr_r_min = params["min_thr_r"]
    thr_r_max = params["max_thr_r"]
    thr_error = params["thr_error"]

    filtered_blobs = []
    blobs_rest = []
    for blob, val in zip(blobs_ext, blobs_params):
        r = blob[2]
        c0 = val[0]
        norm_error = val[3]

        if ((c0 > thr_c0) and (norm_error <= thr_error) and (r <= thr_r_max) and (r >= thr_r_min)):
            filtered_blobs.append(blob)
        else:
            blobs_rest.append(blob)

    return np.array(filtered_blobs), blobs_rest



# -------------DEMO--------------------
if __name__ == "__main__":
    
    from PIL import Image
    import autoscale

    img_path = r"D:\Cloud\Pd_C_0.1%_8mm_0007.tif"


    img = Image.open(img_path).convert('L')
    grayImage = np.array(img, dtype='uint8')    
    currentImage = np.copy(grayImage) 
    
    # lowerBound = autoscale.findBorder(grayImage)        
    # if (lowerBound is not None):
    #     currentImage = currentImage[:lowerBound, :]
    currentImage = grayImage[:890,:]

    params = {
            "sz_med" : 4,   # для предварительной обработки
            "sz_th":  4,    # для предварительной обработки (не надо равное 5 - кружки получаются большие) 
            "thr_br": 10,   # порог яркости для отбрасывания лок. максимумов (Prefiltering)
            "min_dist": 5,  # минимальное расстояние между локальными максимумами при поиске локальных максимумов (Prefiltering)
            "wsize": 9,     # размер окна аппроксимации
            "rs": np.arange(1.0, 7.0, 0.1), # возможные радиусы наночастиц в пикселях
            "best_mode": 3, # выбор лучшей точки в окрестности лок.макс. по norm_error (1 - по с1, 2 - по с0, 3 - по norm_error) 
            "msk": 5,       # берем окошко такого размера с центром в точке локального максимума для уточнения положения наночастицы   
            "met": 'exp',   # аппроксимирующая функция "exp" или "pol" 
            "npar": 2       # число параметров аппроксимации
        }

    # вычисляется только один раз при первом запуске детектирования
    helpMatrs, xy2 = CACHE_HelpMatricesNew(params["wsize"], params["rs"])

    # вычисляется только один раз для одного и тогоже изображения
    lm, currentImage = CACHE_PrefilteringPoints(
        currentImage,
        params["sz_med"],
        params["sz_th"],
        params["min_dist"],
        params["thr_br"]
    )


    BLOBs, BLOBs_params = CACHE_ExponentialApproximationMask_v3(
        currentImage,
        lm,
        xy2,
        helpMatrs,
        params
    )

    params_filter = {
        "thr_c0": 16,
        "min_thr_r": 1,   
        "max_thr_r": 3.6, 
        "thr_error": 0.15, 
    }

    filtered_blobs, blobs_rest = my_FilterBlobs_change(BLOBs, BLOBs_params, params_filter) 
    for i in range(np.shape(BLOBs)[0]): 
        x = BLOBs[i][1]
        y = BLOBs[i][0]
        r = BLOBs[i][2]
        c0 = BLOBs_params[i][0]
        c1 = BLOBs_params[i][1]
        c2 = BLOBs_params[i][2]
        norm_error = BLOBs_params[i][3]
        print("%d (%d, %d): r=%.1f, c0=%.3f, c1=%.3f, c2=%.3f, error=%.4f " % (i, x, y, r, c0, c1, c2, norm_error))

    VisualizationSimpleNums(grayImage, BLOBs, FIGSIZE = (9,7), _show=True, _save = False, outfname='', _dpi=300, lsize=100, lwidth = 1, nums = False)
