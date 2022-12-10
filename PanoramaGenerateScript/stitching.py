import argparse
import logging
import cv2
import os
import numpy as np
from image_stitching import ImageStitcher
from image_stitching import load_frames
from image_stitching import display
import multiprocessing
from functools import partial
#import stitching
__doc__ = '''This script lets us stich images together'''

# utils for our project
def display_alpha(image):
    copied= np.copy(image)
    for i in range(len(copied)):
        for j in range(len(copied[0])):
            copied[i][j]=[copied[i][j][3],copied[i][j][3],copied[i][j][3],copied[i][j][3]]
    cv2.imshow("alphashow", copied)
    cv2.waitKey(0)
def set_alpha_all_zero(image):
    copied= np.copy(image)
    for i in range(len(copied)):
        for j in range(len(copied[0])):
            copied[i][j]=[copied[i][j][0],copied[i][j][1],copied[i][j][2],0]
    return copied
def set_alpha_all_255(image):
    copied= np.copy(image)
    for i in range(len(copied)):
        for j in range(len(copied[0])):
            copied[i][j]=[copied[i][j][0],copied[i][j][1],copied[i][j][2],255]
    return copied
def reverse_alpha(image):
    copied= np.copy(image)
    for i in range(len(copied)):
        for j in range(len(copied[0])):
            if(copied[i][j][3]==0):
                copied[i][j]=[copied[i][j][0],copied[i][j][1],copied[i][j][2],255]
            else:
                copied[i][j]=[copied[i][j][0],copied[i][j][1],copied[i][j][2],0]
    return copied

def parse_args():
    '''parses the command line arguments'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',
                        type=str,
                        nargs='+',
                        help="paths to images, directories, or videos")
    parser.add_argument('--debug', default = False, action='store_true', help='enable debug logging')

    parser.add_argument('--display', default= True, action='store_true', help="display result")
    parser.add_argument('--save', default = True,action='store_true', help="save result to file")
    parser.add_argument("--save-path", default="stitched.png", type=str, help="path to save result")

    return parser.parse_args()

def foreground_generator(pair):
    idx, frame = pair
    subpidx = int(idx//50)
    # if(idx%50==0):
    pathstr ='test2resstep5andsubpanorama\selected2\\'+ str(subpidx).zfill(2)+ ".png"
    print(pathstr)
        # subpidx +=1
    if idx>449:
        print(idx)
        stitcherR = ImageStitcher()
        result=cv2.imread(pathstr,cv2.IMREAD_UNCHANGED)
        # cv2.imshow("res",result)
        # cv2.imshow("frame",frame)
        # cv2.waitKey(0)
        result = set_alpha_all_zero(result)
        stitcherR.add_image(reverse_alpha(frame),mode=1)
        stitcherR.add_image(result,mode=1)
        result = stitcherR.image()
        image_name = f'foreground/foreground_{str(subpidx).zfill(3)}_{str(idx).zfill(3)}.png'
        cv2.imwrite(image_name, result)
        del(stitcherR)


if __name__ == '__main__':
    # args = parse_args()
    args = argparse.ArgumentParser(description=__doc__).parse_args()
    # print(args)
    # args = ["python", "stitching.py", "test2resstep5andsubpanorama/selected"]
    args.paths=['test2resstep5andsubpanorama/selected2']
    #args.paths=['test2data']
    args.debug=False
    args.display=True
    args.save=True
    args.save_path='stitched.png'

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    
    
    stitcher = ImageStitcher()
    images=[]
    for idx, frame in enumerate(load_frames(args.paths)):
        images.append(frame)
    for idx in range(len(images)-1,8,-1):
        frame = images[idx]
        foregroundDirPath=f"foreground/{str(idx).zfill(3)}/"
        paths = os.listdir(foregroundDirPath)
        for i in range(len(paths)):
            paths[i]=foregroundDirPath+paths[i]
        print(paths)
        foregroundimages = []
        for p in paths:
            imageforeground =cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
            imageforeground= cv2.resize(imageforeground,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
            foregroundimages.append([p,imageforeground])

        
        #print(foregroundimages)
        #print("inputheresize")
        #print(np.array(frame).shape)
        #print("testshape")
        #print(frame.shape)
        #print(foregroundimages[0][1].shape)
        stitcher.add_image(frame,foregroundimages=foregroundimages)
        result = stitcher.image()
        fgis = stitcher.foreground()
        fgi = fgis[0][1]
        print(len(fgis),len(fgi),len(fgi[0]),len(result),len(result[1]))
        cv2.imshow("fgi",fgi)
        #cv2.waitKey(0)
        #print("outputheresize")
        #print(result.shape)
        #display_alpha(result)
        if args.display:
            logging.info(f'displaying image {idx}')
            res=result[:,:,0:3]
            display('result', res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        if args.save:
            image_name = f'result_{idx}.png'
            logging.info(f'saving result image on {image_name}')
            cv2.imwrite(image_name, result)
    fgi= stitcher.foreground()
    for item in fgi:
        cv2.imwrite(item[0],item[1])
    
    '''
    #pip install stitching
    images=[]
    mask =[] 
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    for idx, frame in enumerate(load_frames(args.paths)):
        # mask.append(frame[:,:,3])
        #a_mask = frame[:,:,3]
        #a_mask = np.expand_dims(a_mask, axis=2)
        # a_mask[a_mask == 255] = np.array([[255,255,255]])
        #a_mask = np.repeat(a_mask, 3, axis=2)
        # a_mask[a_mask == 0] = 0
        # a_mask = a_mask.astype(int)
        # for i in range(len(frame)):
        #    for j in range(len(frame[0])):
        #        if(frame[i][j][3]==0):
        #            a_mask[i][j] = True
        #mask.append(a_mask)
        img = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
        #image_name = f'test2reducedblackforeground/{idx:03}.png'
        #cv2.imwrite(image_name, frame)
        images.append(img)
    panorama = stitcher.stitch(images)
    image_name = f'finalpanorama.png'
    cv2.imwrite(image_name, panorama[1])   
    ''' 
    '''
    
    stitcher = ImageStitcher()
    subPanoramaIdx=0
    for idx, frame in enumerate(load_frames(args.paths)):
        if(idx%10==0):
            subPanoramaIdx+=1
            del(stitcher)
            stitcher = ImageStitcher()
        #print("inputheresize")
        #print(np.array(frame).shape)
        stitcher.add_image(frame)
        result = stitcher.image()
        #print("outputheresize")
        #print(result.shape)
        #display_alpha(result)
        if args.display:
            logging.info(f'displaying image {idx}')
            res=result[:,:,0:3]
            display('result', res)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        if args.save:
            image_name = f'result_{subPanoramaIdx}_{idx}.png'
            logging.info(f'saving result image on {image_name}')
            cv2.imwrite(image_name, result)
    '''
    ''' 
    if args.save:
        logging.info(f'saving final image to {args.save_path}')
        cv2.imwrite(args.save_path, result)
    '''
    logging.info('finished stitching images together')
    
    '''
    #panorama #mode -1
    image_name = f'finalpanorama.png'
    result=cv2.imread(image_name,cv2.IMREAD_UNCHANGED)
    result= cv2.cvtColor(result,cv2.COLOR_BGR2BGRA)
    cv2.imwrite(image_name, result)
    for idx, frame in enumerate(load_frames(args.paths)):
        stitcherR = ImageStitcher()
        result=cv2.imread(image_name,cv2.IMREAD_UNCHANGED)
        result = set_alpha_all_zero(result)
        stitcherR.add_image(frame)
        stitcherR.add_image(result)
        result = stitcherR.image()
        cv2.imwrite(image_name, result)
        del(stitcherR)
    
    
    '''
    '''
    use_multiprocessing = True
    if use_multiprocessing:
        print("using parallel processing (CPU)")
        pool = multiprocessing.Pool(processes=24)
        parameters_each_iter = []
        frames = []
        for idx, frame in enumerate(load_frames(args.paths)):
            frames.append(frame)
        pool.map(foreground_generator, [(idx, frames[idx]) for idx in range(len(frames))])

    else:
        #render foreground #mode 0 
        subpidx=0
        for idx, frame in enumerate(load_frames(args.paths)):
            pair = (idx, frame)
            foreground_generator(pair)
    '''


    '''
    #all ground,  mode 1
    for idx, frame in enumerate(load_frames(args.paths)):
        stitcherR = ImageStitcher()
        result=cv2.imread(args.save_path,cv2.IMREAD_UNCHANGED)
        #result = set_alpha_all_zero(result)
        frameA = set_alpha_all_255(frame)
        #cv2.imshow("frameA",frameA)
        #cv2.waitKey(0)
        stitcherR.add_image(frameA)
        stitcherR.add_image(result,mode=1)
        result = stitcherR.image()
        image_name = f'allground/allground_{idx}.png'
        cv2.imwrite(image_name, result)
        del(stitcherR)
     '''
    '''
    #No object Video,  mode 2
    subpidx=0
    for idx, frame in enumerate(load_frames(args.paths)):
        print(idx)
        if(idx%50==0):
            pathstr ='test2resstep5andsubpanorama\selected2\\'+ str(subpidx).zfill(2)+ ".png"
            print(pathstr)
            subpidx +=1
        stitcherR = ImageStitcher()
        #print(stitcherR.image().shape)
        result=cv2.imread(pathstr,cv2.IMREAD_UNCHANGED)
        result = set_alpha_all_255(result)
        stitcherR.add_image(result,mode=2)
        stitcherR.add_image(frame,mode=2)
        result = stitcherR.image()
        image_name = f'noobjectvideo/noobjectvideo_{idx}.png'
        cv2.imwrite(image_name, result)
        del stitcherR
    '''
