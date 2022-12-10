import logging

import cv2
import numpy
import numpy as np

__doc__ = '''helper functions for combining images, only to be used in the stitcher class'''


def compute_matches(features0, features1, matcher, knn=5, lowe=0.7):
    '''
    this applies lowe-ratio feature matching between feature0 an dfeature 1 using flann
    '''
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    logging.debug('finding correspondence')

    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

    logging.debug("filtering matches with lowe test")

    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)

    src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive],
                          dtype=numpy.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive],
                          dtype=numpy.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    return src_pts, dst_pts, len(positive)


def combine_images(img0, img1, h_matrix, mode=0,foregroundimages=None,foregroundNew=None):
    '''
    this takes two images and the homography matrix from 0 to 1 and combines the images together!
    the logic is convoluted here and needs to be simplified!
    '''
    logging.debug('combining images... ')

    points0 = numpy.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]],
        dtype=numpy.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]],
        dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))

    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = numpy.concatenate((points0, points2), axis=0)

    [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(numpy.int32)
    [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(numpy.int32)

    h_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    logging.debug('warping previous image...')
    
    output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix),
                                     (x_max - x_min, y_max - y_min))
    if(foregroundimages!=None):
        for i in range(len(foregroundimages)):
            print(foregroundimages[i][0])
            foregroundimages[i][1]=cv2.warpPerspective(foregroundimages[i][1], h_translation.dot(h_matrix),(x_max - x_min, y_max - y_min))
    print(output_img.shape)
    #print(img0.shape)
    #print(-y_min,img0.shape[0] - y_min)
    #print(-x_min,img0.shape[1] - x_min)
    if(mode==2):
        i0=0
        j0=0
        for i in range(-y_min,img0.shape[0] - y_min):
            j0=0
            for j in range(-x_min,img0.shape[1] - x_min):
                #output_img is panorama, img0 is video frame
                if(img0[i0][j0][3]==0 and output_img[i][j][3]==0):
                    img0[i0][j0][3]==255
                elif(img0[i0][j0][3]!=0 and output_img[i][j][3]==0):
                    #output_img[i][j]=img0[i0][j0]
                    pass
                elif(img0[i0][j0][3]==0 and output_img[i][j][3]!=0):
                    img0[i0][j0]=output_img[i][j]
                    pass
                else:
                    img0[i0][j0]=output_img[i][j]
                j0+=1
            i0+=1
        return img0
    else:
        i0=0
        j0=0
        # for i in range(-y_min,img0.shape[0] - y_min):
        #     j0=0
        #     for j in range(-x_min,img0.shape[1] - x_min):
        #         if(img0[i0][j0][3]==0 and output_img[i][j][3]==0):
        #             pass
        #         elif(img0[i0][j0][3]!=0 and output_img[i][j][3]==0):
        #             output_img[i][j]=img0[i0][j0]
        #         elif(img0[i0][j0][3]==0 and output_img[i][j][3]!=0):
        #             pass
        #         else:
        #             if(mode==0):
        #                 output_img[i][j]=img0[i0][j0]
        #             else:
        #                 pass
        #         pass
        #         j0+=1
        #     i0+=1
        
        # accelerated
        '''
        panorama = output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min]
        img = img0
        alpha = img[:,:,3:4]
        #print(alpha.shape)
        alphacopy = np.copy(alpha[:,:,0])
        #print(alphacopy.shape)
        alpha=numpy.insert(alpha,0,alphacopy,axis=2)
        alpha=numpy.insert(alpha,0,alphacopy,axis=2)
        #print(alpha.shape)
        panoramaAlpha = panorama[:,:,3:4]
        #print(panoramaAlpha.shape)
        palphacopy = np.copy(panoramaAlpha[:,:,0])
        #print(palphacopy.shape)
        panoramaAlpha=numpy.insert(panoramaAlpha,0,palphacopy,axis=2)
        panoramaAlpha=numpy.insert(panoramaAlpha,0,palphacopy,axis=2)
        img = img0[:,:,0:3]
        panorama = panorama[:,:,0:3]
        dumy = np.logical_and(panoramaAlpha,alpha)
        panorama = (img*np.logical_and(alpha,True)+panorama*np.logical_and(panoramaAlpha,True))*np.logical_not(dumy) + img*np.logical_and(alpha,True)*dumy
        panoramaAlpha = np.logical_or(panoramaAlpha,alpha)
        panorama=numpy.insert(panorama,3,panoramaAlpha[:,:,0],axis=2)
        output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = panorama
        '''
        
        for i in range(-y_min,img0.shape[0] - y_min):
            j0=0
            for j in range(-x_min,img0.shape[1] - x_min):
                if(img0[i0][j0][3]!=0):
                    if output_img[i][j][3]==0:
                        output_img[i][j]=img0[i0][j0]
                    elif(mode==0):
                        output_img[i][j]=img0[i0][j0]
                j0+=1
            i0+=1
        

        if(foregroundNew!=None):
            for i in range(len(foregroundNew)):
                print(foregroundNew[i][0])
                newforeground = np.zeros_like(output_img)
                newforeground[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = foregroundNew[i][1]
                foregroundimages.append([foregroundNew[i][0],newforeground])
        
        
        '''
        mini= np.amax(np.argmax(output_img[:,:,3],axis=0))
        minj=np.amax(np.argmax(output_img[:,:,3],axis=1))
        output_img = np.flip(output_img,axis=0)
        output_img =np.flip(output_img,axis=1)
        #plot_image(frame3, (10,10))
        maxi= len(output_img)-np.amax(np.argmax(output_img[:,:,3],axis=0))
        maxj = len(output_img[0])-np.amax(np.argmax(output_img[:,:,3],axis=1))
        output_img = np.flip(output_img,axis=0)
        output_img =np.flip(output_img,axis=1)
        '''
        mini = len(output_img)
        minj = len(output_img[0])
        maxi =0
        maxj =0
        for i in range(len(output_img)):
            for j in range(len(output_img[0])):
                if(output_img[i][j][3]!=0):
                    if(mini>i):
                        mini=i
                    if(minj>j):
                        minj=j
                    if(maxi<i):
                        maxi=i
                    if(maxj<j):
                        maxj=j
        #print(output_img.shape)
        #print(mini,minj,maxi,maxj)     
                  
        output_img=output_img[mini:maxi,minj:maxj]
        if(foregroundimages!=None):
            for i in range(len(foregroundimages)):
                foregroundimages[i][1]=foregroundimages[i][1][mini:maxi,minj:maxj]
        return output_img
    #print(img0.shape)
    #print(-y_min,img0.shape[0] - y_min)
    #print(-x_min,img0.shape[1] - x_min)
    #output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    
