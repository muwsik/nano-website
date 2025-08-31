# -*- coding: cp1251 -*-

import numpy as np
from skimage import morphology
import cv2
from skimage.feature import peak_local_max

import streamlit as st

from joblib import Parallel, delayed
from functools import partial

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

def FindAreasToDelete(img, threshLines, max_area): 
    #ret_otsu, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    #fig, ax = plt.subplots(1, 1, figsize=(9,7), sharex=True, sharey=True)
    #ax.imshow(otsu_img, cmap='gray')
    #print('otsu:', ret_otsu)
    
    #fig, ax = plt.subplots(1, 1, figsize=(9,7), sharex=True, sharey=True)
    #ax.imshow(img, cmap='gray')
    
    img_th_gamma = (np.around(np.power(img / np.max(img), 0.4) * 255)).astype(np.uint8)
    #fig, ax = plt.subplots(1, 1, figsize=(9,7), sharex=True, sharey=True)
    #ax.imshow(img_th_gamma, cmap='gray')
    
    #ret_otsu2, otsu_img2 = cv2.threshold(img_th_gamma, 0, 255, cv2.THRESH_OTSU)
    #fig, ax = plt.subplots(1, 1, figsize=(9,7), sharex=True, sharey=True)
    #ax.imshow(otsu_img2, cmap='gray')
    #print('otsu2:', ret_otsu2)
    
    #ret, thresh_img = cv2.threshold(img, threshLines, 255, cv2.THRESH_BINARY)
    ret, thresh_img = cv2.threshold(img_th_gamma, threshLines, 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    i = 0
    big_contours = []
    for contour in contours:
        i = i + 1
        area = cv2.contourArea(contour)
        if area > max_area:
            big_contours.append(contour)
        
        img_contours = np.uint8(np.zeros((img.shape[0], img.shape[1])))
        cv2.drawContours(img_contours, big_contours, thickness=-1, color=(255, 255, 255), contourIdx=-1)
        #cv2.drawContours(img_contours, big_contours, thickness=-1, color=(255, 0, 0), contourIdx=-1)
        cv2.drawContours(img, big_contours, thickness=-1, color=(255, 0, 0), contourIdx=-1)
    return img, img_contours, big_contours

def DeleteBorderPoints(blobs, _img_contours, value):
    blobs_corr = []
    for i in range(len(blobs)):
        if _img_contours[int(blobs[i][0])][int(blobs[i][1])] != value:
            blobs_corr.append(blobs[i])
    return blobs_corr.copy()

@st.cache_data(show_spinner = False, max_entries = 5)
def ApproximationMain(c_img, c_lmblobs, params, c_min_dist, c_check):
    # _check ==True - добавление точки только если нет близкой к ней, иначе - добавление без проверки
    shifts_x = [-0.5, 0, 0.5]
    shifts_y = shifts_x
    _blobs_ = []
    for i in range(np.shape(c_lmblobs)[0]):
        x = c_lmblobs[i][0]
        y = c_lmblobs[i][1]
        #print(" ---- local max %i (%i, %i)---- " % (i, x,y))  
        success, subimg = GetSubimage(c_img, (x, y), params["wsize"])
        if success:     
            subimg[subimg==0] = 1                           
            x_new, y_new, shiftx, shifty, c0, c1, norm_error, sq_er = \
                ApproximationWithFindingTheBestCenter_Shift(c_img, (x,y), params, False, shifts_x, shifts_y)
            if c1<0: 
                r = 0
            else: 
                r = 1 / np.sqrt(c1)
            blob1 = (x_new-shiftx,y_new-shifty, r, c0,c1, norm_error, sq_er)
            if c_check:
                _blobs_ = CheckAndAppendBlob(_blobs_, blob1, c_min_dist, 5)
            else: 
                _blobs_.append(blob1)
            #print('npoints = ', np.shape(_blobs_)[0])
        else:
            pass#print('near bound point')
    #_blobs_shift = BlobsApplyShift(_blobs_appr)
    return _blobs_.copy()

def ApproximationWithFindingTheBestCenter_Shift(img, point, params, prn, shifts_x, shifts_y):
    # рассматриваем точку локального максимума и точки вокруг него 
    # из окошка размером params["msk"] 
    # для уточнения координат центра наночастицы, который не всегда совпадает
    # с локальным максимумом
    # всего npnts=msk*msk точек,  ppx, ppy - список координат этих точек
    # params["best_mode"] = 1 или 2: среди точек, удовлетворяющих условиям, выбираем наилучшую 
    # по c0 (best_mode = 1), по с1 (best_mode = 2) или по нормированной ошибке (иначе)
    # params["npar"] = 3 # число параметров аппроксимации 3 или 2
    
    msk = params["msk"]
    wsize = params["wsize"]
    best_mode = params["best_mode"]
    
    ppx, ppy = ExtractPointsBySquareMask(point, np.shape(img), msk)
    npoints =np.shape(ppx)[0]  

    first = True
    for pp in range(npoints):
        #print("  -----  pp=%i (%d, %d)" % (pp, ppx[pp],ppy[pp]))
        # вырезаем окошко subimg из изображения img
        #print(' \n * pp ', pp,  (ppx[pp],ppy[pp]))
        #success, subimg = GetSubimageNew(img, (ppx[pp],ppy[pp]), wsize)        
        success, subimg = GetSubimage(img, (ppx[pp],ppy[pp]), wsize)        
        if success:
            #if (prn):
                #print('\n success ', np.shape(subimg))
                #print(subimg)
            # непосредственно аппроксимация
            #(ind, shiftx, shifty, c0, c1, norm_error, rel_brdif) = ApproxInWindow_StrictCalc_Shift(wsize, subimg, shifts_x, shifts_y)           
            (ind, shiftx, shifty, c0, c1, norm_error, sq_er) = ApproxInWindow_StrictCalc_Shift(wsize, subimg, shifts_x, shifts_y)           
            if c1<0:
                r = 0
            else:
                r = 1 / np.sqrt(c1)
            if(prn):
                #print("     %d (%.1f, %.1f): x=%d, y=%d, shx=%.1f, shy=%.1f, r=%.1f, c0=%.3f, c1=%.3f, error=%.4f, rel_brdif=%.3f " % (pp, ppx[pp]-shiftx, ppy[pp]-shifty, ppx[pp], ppy[pp], shiftx, shifty, r, c0, c1, norm_error, rel_brdif))
                print("     %d (%.1f, %.1f): x=%d, y=%d, shx=%.1f, shy=%.1f, r=%.1f, c0=%.3f, c1=%.3f, error=%.4f, sq_er=%.3f " % (pp, ppx[pp]-shiftx, ppy[pp]-shifty, ppx[pp], ppy[pp], shiftx, shifty, r, c0, c1, norm_error, sq_er))
            if first: 
                ibest = pp
                x = ppx[ibest]
                y = ppy[ibest]
                optc0 = c0
                optc1 = c1
                opt_shiftx = shiftx
                opt_shifty = shifty
                opt_error = norm_error
                opt_sq_er = sq_er
            first = False    
            is_better = False
            match best_mode:
                case 1:
                    if c1>optc1: 
                        is_better = True
                case 2:
                    if c0>optc0:
                        is_better = True
                case 3:
                    if norm_error<opt_error:
                        is_better = True
                case 4:
                    if sq_er<opt_sq_er:
                        is_better = True
            if is_better:        
                ibest = pp
                opt_error = norm_error
                optc0 = c0
                optc1 = c1
                opt_shiftx = shiftx
                opt_shifty = shifty
                x = ppx[ibest]
                y = ppy[ibest] 
                #opt_rel_brdif = rel_brdif
                opt_sq_er = sq_er
            
                
        else:
            if(prn):
                print("     %d (%d, %d): near border point" % (pp, ppx[pp], ppy[pp]))
    #return x, y, opt_shiftx, opt_shifty, optc0, optc1, opt_error, opt_rel_brdif    
    return x, y, opt_shiftx, opt_shifty, optc0, optc1, opt_error, opt_sq_er  

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

def ApproxInWindow_StrictCalc_Shift(wsize, z, shifts_x, shifts_y):
    """Аппроксимация функции z экспоненциальной функцией
    c0*exp(-c1(x^2+y^2))
    со смещениями центра на shifts_x по х и shifts_y по y (без изменения окна!) 
    и выбором лучшего варианта по значению нормированной ошибки
    z  [wsize x wsize] - матрица значений аппроксимируемой функции
    оптимальные значения параметров вычисляются напрямую (!!!) как решение СЛАУ 
    """
    if (z == 0).all():
        return 0, 0, 0
    
    hsz = wsize // 2  # max and min values of positive and negative coordinate values
    x = np.array((np.arange(-hsz, hsz+1)))
    y = x
    z_corr = z.copy()
    z_corr[np.where(z==0)]=1 # !!! принудительно убираем нули для корректного логарифмирования 
    
    npoints = wsize*wsize
    lnz = np.log(z_corr)

    smlnz = np.sum(lnz)
    
    # набор смещений по каждой координате: 
    i = 0
    res_matr = np.zeros((9,7))
    for shiftx in shifts_x:
        xx = (x+shiftx)*(x+shiftx)
        for shifty in shifts_y: 
            yy = (y+shifty)*(y+shifty)
            xy2 = np.array([xx]*wsize).T + np.array([yy]*wsize)
            smxy2 = np.sum(xy2)
            chisl = npoints*np.sum(lnz*xy2) - smxy2*smlnz
            znam = smxy2*smxy2 - npoints*np.sum(xy2*xy2)
            b = chisl/znam
            a = np.exp((b*smxy2+smlnz)/npoints)
            zapr = a * np.exp(-b * xy2)        
            delta = z - zapr 
            chisl = np.sqrt( sum( sum(np.array(delta) * delta) )/npoints)
            znam1 = np.sqrt( (sum( sum(np.array(z, dtype='float64')*z) ))/npoints)
            znam2 = np.sqrt( (sum( sum(np.array(zapr)*zapr) ))/npoints)        
            norm_error = chisl / ( znam1 +znam2 ) 
            #rel_brdif = np.abs((z[hsz+1, hsz+1]-a))/z[hsz+1, hsz+1]
            br = z[hsz+1, hsz+1]
            sq_err = chisl
            #res_matr[i,:] = [i, shiftx, shifty, a, b, norm_error, rel_brdif]
            res_matr[i,:] = [i, shiftx, shifty, a, b, norm_error, sq_err]
            i = i + 1
    #best_row = np.argmin(res_matr[:,5])
    best_row = np.argmin(res_matr[:,6]) # ! минимум по обычной среднеквадратической ошибке
    
    #print('res_matr:\n', res_matr)
    #print('   *  shift:',res_matr[best_row])
    return res_matr[best_row]

def CheckAndAppendBlob(_blobs, blob_new, _min_dist, index):
    #blob1 = (x_new,y_new, r, c0,c1, norm_error, sq_er)
    # проверка существования в списке такой же или близкой частицы
    app = True
    nbl = np.shape(_blobs)[0]
    for i in range(nbl):
        blob = _blobs[i]
        dx = blob_new[0]-blob[0]
        dy = blob_new[1]-blob[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist<=_min_dist: 
            #print('i=', i, '(', blob[0],blob[1],')', '(',blob_new[0], blob_new[1],')', 'dist=',dist, '<=min_dist=', _min_dist)
            if blob_new[index]<blob[index]:
                #print(blob_new[index],'<',blob[index])
                _blobs[i] = blob_new
            app = False
            break
    if app:
        _blobs.append(blob_new)
    return _blobs

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