import cv2
import numpy as np

def drawCircle(image,p,color):
    cv2.circle(image,p,15,color=color,thickness=-1)

def drawPanorama(imageNext,p,image,d):
    p1,p2,p3,p4 = p
    d1,d2,d3,d4 = d
    height,width = image.shape[:2]
    #print(height,width)
    imageNext = cv2.copyMakeBorder(imageNext, 0 , height-imageNext.shape[0],0, width-imageNext.shape[1] , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    #heights,widths = imageNext.shape[:2]
    #print(heights,widths)
    
    copied = np.copy(image)
    tlp=[[0],[0],[1]]
    trp = [[width],[0],[1]]
    blp=[[0],[height],[1]]
    brp=[[width],[height],[1]]
    borderPoint= [tlp,trp,blp,brp]
    '''
    for p in borderPoint:
        #point = (p[0][0],p[1][0])
        #drawCircle(copied,point,color=(255,0,0))
        pass
    '''
    for p in [p1,p2,p3,p4]:
        drawCircle(imageNext,p,color=(0,0,255))
        pass

    for p in [d1,d2,d3,d4]:
        drawCircle(copied,p,color=(0,255,0))
        pass
    
    #cv2.namedWindow("image", cv2.WINDOW_NORMAL)    
    #cv2.imshow("image",copied)
    #cv2.namedWindow("imageNext", cv2.WINDOW_NORMAL)    
    #cv2.imshow("imageNext", imageNext)
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
    #return 0
    src = np.float32([p1,p2,p3,p4])
    des = np.float32([d1,d2,d3,d4])

    transformMatrix= cv2.getPerspectiveTransform(src,des)

    #print(transformMatrix)
    xborder = [0,width]
    yborder = [0,height]
    for p in borderPoint:
        tp = np.dot(transformMatrix,np.float32(p))
        tp = tp/tp[2][0]
        #print(tp)
        xborder.append(tp[0][0])
        yborder.append(tp[1][0])
        #drawCircle(correctedImage,(int(tp[0][0]),int(tp[1][0])),color=(255,0,0))
    #print(xborder,yborder)
    #print(min(xborder), (max(xborder)-width), min(yborder), max(yborder)-height)
    copied = cv2.copyMakeBorder(copied, int(abs(min(yborder))), int(abs(max(yborder)-height)),int(abs(min(xborder))),int(abs(max(xborder)-width)) , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    imageNext = cv2.copyMakeBorder(imageNext, int(abs(min(yborder))), int(abs(max(yborder)-height)),int(abs(min(xborder))),int(abs(max(xborder)-width)) , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    #print(copied.shape[:2])
    p1=(p1[0]+int(abs(min(xborder))),p1[1]+int(abs(min(yborder))))
    p2=(p2[0]+int(abs(min(xborder))),p2[1]+int(abs(min(yborder))))
    p3=(p3[0]+int(abs(min(xborder))),p3[1]+int(abs(min(yborder))))
    p4=(p4[0]+int(abs(min(xborder))),p4[1]+int(abs(min(yborder))))
    d1=(d1[0]+int(abs(min(xborder))),d1[1]+int(abs(min(yborder))))
    d2=(d2[0]+int(abs(min(xborder))),d2[1]+int(abs(min(yborder))))
    d3=(d3[0]+int(abs(min(xborder))),d3[1]+int(abs(min(yborder))))
    d4=(d4[0]+int(abs(min(xborder))),d4[1]+int(abs(min(yborder))))
    '''
    for p in [p1,p2,p3,p4]:
        drawCircle(copied,p,color=(0,0,255,255))
        pass
    for p in [d1,d2,d3,d4]:
        drawCircle(copied,p,color=(255,255,0,255))
        pass
    '''
    src = np.float32([p1,p2,p3,p4])
    des = np.float32([d1,d2,d3,d4])
    #print(src,des)
    transformMatrix= cv2.getPerspectiveTransform(src,des)
    height,width = imageNext.shape[:2]
    correctedImage = cv2.warpPerspective(imageNext,transformMatrix,(width,height),borderValue=(0,0,0,0))
    difference = 0 
    countdiff=0
    #print(1)
    for w in range(width):
        for h in range(height):
            if(correctedImage[h][w][3]!=0):
                if(copied[h][w][3]!=0):
                    copied[h][w]=copied[h][w]//[2,2,2,2]+correctedImage[h][w]//[2,2,2,2]
                    countdiff+=1
                    difference = sum([abs(int(copied[h][w][i])-int(correctedImage[h][w][i])) for i in range(3)])
                else:
                    copied[h][w]=correctedImage[h][w]
                    
    difference/=countdiff
    #print(difference)            
    #print(2)
    #copied.paste(correctedImage,(0,0))
    #cv2.namedWindow("copied", cv2.WINDOW_NORMAL)    
    #cv2.imshow("copied",copied)
    #cv2.namedWindow("imageNext", cv2.WINDOW_NORMAL)    
    #cv2.imshow("imageNext", imageNext)
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
    return copied, difference


def getTransformMatrix(p1,p2,p3,p4,d1,d2,d3,d4):
    src = np.float32([p1,p2,p3,p4])
    des = np.float32([d1,d2,d3,d4])
    return cv2.getPerspectiveTransform(src,des)
def chooseCandidate(ps,ds):
    panoramas=[]
    diffs=[]
    for i in range(len(ps)):
        print("calculate panorama candidate "+str(i))
        panorama, diff = drawPanorama(image,ps[i],prePanorama,ds[i])
        panoramas.append(panorama)
        diffs.append(diff)
    print("differences of candidate panorama"+str(diffs))
    finalpanorama = panoramas[diffs.index(min(diffs))]
    cv2.namedWindow("finalpanorama", cv2.WINDOW_NORMAL)    
    cv2.imshow("finalpanorama",finalpanorama)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
    return finalpanorama
        
        
p1=(1053,570)
p2=(463,570)
p3=(810,180)
p4=(35,180)
d1=(1628,570)
d2=(1038,570)
d3=(1385,180)
d4=(610,180)  

ps=[[[1053,570],[463,570],[810,180],[35,180]],[[1055,570],[463,572],[810,181],[35,182]]]
ds=[[[1628,570],[1038,570],[1385,180],[610,180]],[[1628,572],[1038,571],[1387,180],[615,180]]]
path=""
prePanorama = cv2.imread(path+"prePonorama.png")
prePanorama = cv2.cvtColor(prePanorama, cv2.COLOR_RGB2RGBA)
image = cv2.imread(path+"image.png")
image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
chooseCandidate(ps,ds)