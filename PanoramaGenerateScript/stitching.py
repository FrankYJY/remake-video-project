import argparse
import logging
import cv2
import numpy as np
from image_stitching import ImageStitcher
from image_stitching import load_frames
from image_stitching import display

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


if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    '''
    stitcher = ImageStitcher()

    for idx, frame in enumerate(load_frames(args.paths)):
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
            image_name = f'result_{idx}.png'
            logging.info(f'saving result image on {image_name}')
            cv2.imwrite(image_name, result)
    if args.save:
        logging.info(f'saving final image to {args.save_path}')
        cv2.imwrite(args.save_path, result)
    '''
    #render foreground
    for idx, frame in enumerate(load_frames(args.paths)):
        stitcherR = ImageStitcher()
        result=cv2.imread(args.save_path,cv2.IMREAD_UNCHANGED)
        result = set_alpha_all_zero(result)
        stitcherR.add_image(reverse_alpha(frame))
        stitcherR.add_image(result)
        result = stitcherR.image()
        image_name = f'foreground/foreground_{idx}.png'
        cv2.imwrite(image_name, result)
        del(stitcherR)
    
    #all ground
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
    logging.info('finished stitching images together')
    
