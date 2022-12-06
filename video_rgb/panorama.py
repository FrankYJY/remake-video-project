import cv2
import numpy as np

def drawCircle(image,p,color):
    cv2.circle(image,p,40,color=color,thickness=-1)

def drawPanorama(imageNext,p,image,d):
    p1,p2,p3,p4 = p
    d1,d2,d3,d4 = d
    #imageNext is bigger and is panorama
    height,width = image.shape[:2]
    #print(height,width)
    heightN,widthN = imageNext.shape[:2]
    maxHeight=max(height,heightN)
    maxWidth = max(width,widthN)
    #height = maxHeight
    #width = maxWidth
    image = cv2.copyMakeBorder(image, 0 , max(0,maxHeight-image.shape[0]),0, max(0,maxWidth-image.shape[1]) , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    imageNext = cv2.copyMakeBorder(imageNext, 0 , max(0,maxHeight-imageNext.shape[0]),0, max(0,maxWidth-imageNext.shape[1]) , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    #print("aaaa")
    #print(image.shape[:2])
    #print(imageNext.shape[:2])
    #print(heights,widths)
    
    copied = np.copy(image)
    #cv2.namedWindow("image", cv2.WINDOW_NORMAL)    
    #cv2.imshow("image",copied)
    #cv2.namedWindow("imageNext", cv2.WINDOW_NORMAL)    
    #cv2.imshow("imageNext", imageNext)
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
    
    tlp=[[0],[0],[1]]
    trp = [[widthN],[0],[1]]
    blp=[[0],[heightN],[1]]
    brp=[[widthN],[heightN],[1]]
    borderPoint= [tlp,trp,blp,brp]
    '''
    for p in borderPoint:
        #point = (p[0][0],p[1][0])
        #drawCircle(copied,point,color=(255,0,0))
        pass
    
    for p in [p1,p2,p3,p4]:
        drawCircle(imageNext,p,color=(0,0,255))
        pass

    for p in [d1,d2,d3,d4]:
        drawCircle(copied,p,color=(0,255,0))
        pass
    '''
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
    xborder = [0,maxWidth]
    yborder = [0,maxHeight]
    for p in borderPoint:
        tp = np.dot(transformMatrix,np.float32(p))
        tp = tp/tp[2][0]
        #print(tp)
        xborder.append(tp[0][0])
        yborder.append(tp[1][0])
        #drawCircle(correctedImage,(int(tp[0][0]),int(tp[1][0])),color=(255,0,0))
    #print(xborder,yborder)
    #print(min(xborder), (max(xborder)-width), min(yborder), max(yborder)-height)
    #print("MakeBorder Size:")
    #print(copied.shape[:2])
    copied = cv2.copyMakeBorder(copied, int(abs(min(yborder))), int(abs(max(yborder)-maxHeight)),int(abs(min(xborder))),int(abs(max(xborder)-maxWidth)) , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    imageNext = cv2.copyMakeBorder(imageNext, int(abs(min(yborder))), int(abs(max(yborder)-maxHeight)),int(abs(min(xborder))),int(abs(max(xborder)-maxWidth)) , cv2.BORDER_CONSTANT, value=[0, 0, 0,0])
    centerPoint= [int(abs(min(xborder))),int(abs(min(yborder)))] 
    #print("aaaa")
    #print(yborder,xborder)
    #print(copied.shape[:2])
    #print(imageNext.shape[:2])
    #cv2.namedWindow("image", cv2.WINDOW_NORMAL)    
    #cv2.imshow("image",copied)
    #cv2.namedWindow("imageNext", cv2.WINDOW_NORMAL)    
    #cv2.imshow("imageNext", imageNext)
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
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
    correctedImage = cv2.warpPerspective(imageNext,transformMatrix,(width,height),borderValue=(0,0,0,0),flags=cv2.INTER_NEAREST)
    difference = 0 
    countdiff=0
    #print(1)
    minh=height
    maxh=0
    minw=width
    maxw=0
    for w in range(width):
        for h in range(height):
            if(correctedImage[h][w][3]!=0):
                if(copied[h][w][3]!=0):
                    countdiff+=1
                    difference = sum([abs(int(copied[h][w][i])-int(correctedImage[h][w][i])) for i in range(3)])
                    #copied[h][w]=copied[h][w]//[2,2,2,2]+correctedImage[h][w]//[2,2,2,2]
                    #copied[h][w]=copied[h][w]
                    copied[h][w]=copied[h][w]
                else:
                    copied[h][w]=correctedImage[h][w]
            if(copied[h][w][3]!=0):
                if(h<minh):
                    minh=h
                if(h>maxh):
                    maxh=h
                if(w<minw):
                    minw=w
                if(w>maxw):
                    maxw=w
    copied = np.array([copied[x][minw:maxw] for x in range(minh,maxh)])            
    centerPoint= [int(abs(min(xborder)))-minh,int(abs(min(yborder)))-minw]                
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
    print(centerPoint)
    return copied, difference,centerPoint,transformMatrix


def getTransformMatrix(p1,p2,p3,p4,d1,d2,d3,d4):
    src = np.float32([p1,p2,p3,p4])
    des = np.float32([d1,d2,d3,d4])
    return cv2.getPerspectiveTransform(src,des)
def generatePanoramaCandidate(image,ps,nowPanorama,ds,tfm):
    panoramas=[]
    diffs=[]
    centerPoints=[]
    transformMatrixs=[]
    for i in range(len(ps)):
        print("calculate panorama candidate "+str(i))
        panorama, diff, centerPoint,transformMatrix = drawPanorama(nowPanorama,ds[i],image,ps[i])
        panoramas.append(panorama)
        diffs.append(diff)
        centerPoints.append(centerPoint)
        transformMatrixs.append(transformMatrix)
    print("differences of candidate panorama"+str(diffs))
    finalpanorama = panoramas[diffs.index(min(diffs))]
    finalCenterPoint = centerPoints[diffs.index(min(diffs))]
    finalTransformMatrix = transformMatrixs[diffs.index(min(diffs))]
    tfm.append(finalTransformMatrix)
    print(str(diffs.index(min(diffs)))+" selected")
    #cv2.namedWindow("finalpanorama", cv2.WINDOW_NORMAL)    
    #cv2.imshow("finalpanorama",finalpanorama)
    #cv2.waitKey(0) # waits until a key is pressed
    #cv2.destroyAllWindows() # destroys the window showing image
    return finalpanorama,finalCenterPoint,tfm
def getFourPoints(motion_vector_matrix,extracted_frame_n1_RGBA_s,block_size,borderSize,centerPoint):
    #resp=[]
    #resd=[]
    background = extracted_frame_n1_RGBA_s[0]
    #meany = 0
    #meanx = 0
    #count=0
    allBackgroundPos=[]
    top=[len(motion_vector_matrix),0]
    bottom=[0,len(motion_vector_matrix[0])]
    left=[len(motion_vector_matrix),len(motion_vector_matrix[0])]
    right=[0,0]
    topd=[len(motion_vector_matrix),0]
    bottomd=[0,len(motion_vector_matrix[0])]
    leftd=[len(motion_vector_matrix),len(motion_vector_matrix[0])]
    rightd=[0,0]
    for h in range(borderSize,len(motion_vector_matrix)-borderSize):
        for w in range(borderSize,len(motion_vector_matrix[0])-borderSize):
            #count +=1
            #meany += motion_vector_matrix[h][w][0][0]
            #meanx += motion_vector_matrix[h][w][0][1]
            center = [h*block_size+block_size//2,w*block_size+block_size//2]
            if(background[center[0]][center[1]][3]!=0):
                if(top[0]>h):
                    top =[h,w]
                    topd = [motion_vector_matrix[h][w][0][0],motion_vector_matrix[h][w][0][1]]
                if(bottom[0]<=h):
                    bottom = [h,w]
                    bottomd = [motion_vector_matrix[h][w][0][0],motion_vector_matrix[h][w][0][1]]
                if(left[1]>=w):
                    left = [h,w]
                    leftd = [motion_vector_matrix[h][w][0][0],motion_vector_matrix[h][w][0][1]]
                if(right[1]<w):
                    right = [h,w]
                    rightd = [motion_vector_matrix[h][w][0][0],motion_vector_matrix[h][w][0][1]]
                #[motion_vector_matrix[h][w][0][0]],[motion_vector_matrix[h][w][0][0]]  motion vector
    #meany /= count
    #meanx /=count
    top = [top[1]*block_size+block_size//2,top[0]*block_size+block_size//2]
    bottom = [bottom[1]*block_size+block_size//2,bottom[0]*block_size+block_size//2]
    left = [left[1]*block_size+block_size//2,left[0]*block_size+block_size//2]
    right =[right[1]*block_size+block_size//2,right[0]*block_size+block_size//2]
    #print([top,bottom,left,right])
    #print([topd,bottomd,leftd,rightd])
    direction=-1
    if(direction==0):
        topd=[top[0]+topd[1]+centerPoint[0],top[1]+topd[0]+centerPoint[1]]
        bottomd=[bottom[0]+bottomd[1]+centerPoint[0],bottom[1]+bottomd[0]+centerPoint[1]]
        leftd=[left[0]+leftd[1]+centerPoint[0],left[1]+leftd[0]+centerPoint[1]]
        rightd=[right[0]+rightd[1]+centerPoint[0],right[1]+rightd[0]+centerPoint[1]]
    else:
        topd=[top[0]-topd[1]+centerPoint[0],top[1]-topd[0]+centerPoint[1]]
        bottomd=[bottom[0]-bottomd[1]+centerPoint[0],bottom[1]-bottomd[0]+centerPoint[1]]
        leftd=[left[0]-leftd[1]+centerPoint[0],left[1]-leftd[0]+centerPoint[1]]
        rightd=[right[0]-rightd[1]+centerPoint[0],right[1]-rightd[0]+centerPoint[1]]
    
    #resp.append([top,bottom,left,right])
    #resd.append([topd,bottomd,leftd,rightd])
    
    return [top,bottom,left,right],[topd,bottomd,leftd,rightd]
                
    
if __name__ == "__main__":
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
    ps=[[[1053,570],[463,570],[810,180],[35,180]]]
    ds=[[[1628,570],[1038,570],[1385,180],[610,180]]]
    path=""
    prePanorama = cv2.imread(path+"prePonorama.png")
    prePanorama = cv2.cvtColor(prePanorama, cv2.COLOR_RGB2RGBA)
    image = cv2.imread(path+"image.png")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    print(prePanorama.shape[:2])
    print(image.shape[:2])
    finalpanorama,finalCenterPoint,TransformMatrix = generatePanoramaCandidate(image,ps,prePanorama,ds,[])
    cv2.namedWindow("finalpanorama", cv2.WINDOW_NORMAL)    
    cv2.imshow("finalpanorama",finalpanorama)
    cv2.waitKey(0)
    

