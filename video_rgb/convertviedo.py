import numpy as np
import cv2
import os
import collections
#input the video path
#output an array [number of frame]*[h]*[w]*[4] BGRA and the fps of the video
def convert_video_2_bgra(video_path):

    cap = cv2.VideoCapture(video_path)
    arr = []
    if not cap.isOpened():
        print("Error: video can not load")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        arr.append(bgra)
        #arr.append(frame)
        #if(cv2.waitKey(1)==ord('q')):
          #  break
    cap.release()
    #cv2.destroyAllWindows()

    return (arr,fps)

#input an array [number of frame]*[h]*[w]*[4] BGRA and the fps of the video
#output a video at the local
def convert_bgra_2_video(arr, fps):

    height, width = len(arr[0]), len(arr[0][0])
    out = cv2.VideoWriter("new_video1.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    for i in range(len(arr)):
        out.write(cv2.cvtColor(arr[i], cv2.COLOR_BGRA2BGR))
    out.release()


def read_file_of_res_run_app_one(res_path, foreground_frame_num):
    background_path = res_path + "/stitched.png"
    foreground_path = res_path + "/foreground/foreground_"
    background_pic = cv2.imread(background_path)
    #background_pic = cv2.cvtColor(background_pic, cv2.COLOR_BGR2BGRA)
    foreground_frame_arr = []
    for dir_index in range(foreground_frame_num):
        cur_path = foreground_path+str(dir_index)+".png"
        foreground_pic = cv2.imread(cur_path,cv2.IMREAD_UNCHANGED)
        #foreground_pic = cv2.cvtColor(foreground_pic, cv2.COLOR_BGR2BGRA)
        foreground_frame_arr.append(foreground_pic)
    app_one_display_trails(foreground_frame_arr, 2, background_pic)

#input any object foreground frame -- foreground_frame_arr [number of frame][h][w][4]
#input the number of elements of the same object show in trail and background_panorama [h][w][4]
#output the object display trail "app_one.png"

def app_one_display_trails(foreground_frame_arr, foreground_element_num, background_panorama):

    display_trails_pic = background_panorama
    display_trails_pic_height = len(display_trails_pic)
    display_trails_pic_width = len(display_trails_pic[0])
    interval_between_frames = (int)((len(foreground_frame_arr)-1)/(foreground_element_num-1))
    for i in range(foreground_element_num):
        for h_index in range(display_trails_pic_height):
            for w_index in range(display_trails_pic_width):
                #load alpha data of foreground_frame_arr. if it is 255 replace it with the display_trails_pic.
                if foreground_frame_arr[i*interval_between_frames][h_index][w_index][3] == 255\
                    and foreground_frame_arr[i*interval_between_frames][h_index][w_index][0] != 0\
                    and foreground_frame_arr[i*interval_between_frames][h_index][w_index][1] != 0\
                    and foreground_frame_arr[i*interval_between_frames][h_index][w_index][2] != 0:
                        display_trails_pic[h_index][w_index][0] = foreground_frame_arr[i*interval_between_frames][h_index][w_index][0]
                        display_trails_pic[h_index][w_index][1] = foreground_frame_arr[i*interval_between_frames][h_index][w_index][1]
                        display_trails_pic[h_index][w_index][2] = foreground_frame_arr[i*interval_between_frames][h_index][w_index][2]
    save_pic_status = cv2.imwrite('app_one.png', display_trails_pic)
    print("App one Image written save file: ", save_pic_status)

def read_file_of_res_run_app_two(res_path, foreground_frame_num, fps,  height, width, path, test_mode):
    background_path = res_path + "/stitched.png"
    foreground_path = res_path + "/foreground/foreground_"
    background_pic = cv2.imread(background_path)
    #background_pic = cv2.cvtColor(background_pic, cv2.COLOR_BGR2BGRA)
    foreground_frame_arr = []
    for dir_index in range(foreground_frame_num):
        cur_path = foreground_path+str(dir_index)+".png"
        foreground_pic = cv2.imread(cur_path,cv2.IMREAD_UNCHANGED)
        #foreground_pic = cv2.cvtColor(foreground_pic, cv2.COLOR_BGR2BGRA)
        foreground_frame_arr.append(foreground_pic)
    app_two_create_video(foreground_frame_arr, background_pic, fps, height, width, path, test_mode)

#input ALL objects foreground frame -- foreground_frame_arr [number of frame][h][w][4]
#input location_path is a n*3 array which [n][y, x, time] foreground_frame_arr have all foreground
#input the output video info fps, video height, video width
#output a video that follow the path
def app_two_create_video(foreground_frame_arr, background_panorama, fps, video_height, video_width, location_path, test_mode):
    #from left to right
    if test_mode:
        back_width = len(background_panorama[0])
        end_time = (int) (len(foreground_frame_arr)/fps)
        location_path = [[0,back_width-video_width,0],[0,0,end_time]]

    frame_arr = []
    start_frame_location = location_path[0]
    for path_index in range(1,len(location_path)):
        next_frame_location = location_path[path_index]
        number_of_frame_between_interval = (int)(next_frame_location[2] - start_frame_location[2]) * fps
        height_change = (next_frame_location[0] - start_frame_location[0])/number_of_frame_between_interval
        height_change = (int) (height_change)
        width_change = (next_frame_location[1] - start_frame_location[1])/number_of_frame_between_interval
        width_change = (int) (width_change)
        for frame_index in range(number_of_frame_between_interval):
            display_pic = np.zeros((video_height, video_width, 4))
            display_pic = display_pic.astype('uint8')
            for h_index in range(video_height):
                for w_index in range(video_width):
                    if foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][3] == 255:
                        display_pic[h_index][w_index][0] = foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][0]
                        display_pic[h_index][w_index][1] = foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][1]
                        display_pic[h_index][w_index][2] = foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][2]
                        display_pic[h_index][w_index][3] = 255
                    else:
                        display_pic[h_index][w_index][0] = background_panorama[start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][0]
                        display_pic[h_index][w_index][1] = background_panorama[start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][1]
                        display_pic[h_index][w_index][2] = background_panorama[start_frame_location[0]+height_change*frame_index+h_index][start_frame_location[1]+width_change*frame_index+w_index][2]
                        display_pic[h_index][w_index][3] = 255
            frame_arr.append(display_pic)
        start_frame_location = next_frame_location
    convert_bgra_2_video(frame_arr, fps)

#combine object foreground frame
#input foreground_frame_object_arr [num of object][number of frame][h][w][4]
#input selected objects array if ith object selected, i-1th index is 1 otherwise 0 [1,1,1...0,1,0]
#output foreground_frame_arr [number of frame][h][w][4]
def combine_object_foreground_frame(foreground_frame_object_arr, selected_objects_arr, background_frame_object_arr):
    foreground_frame_object_arr_height = len(foreground_frame_object_arr[0])
    foreground_frame_object_arr_width = len(foreground_frame_object_arr[0][0])
    #foreground_frame_arr = np.zeros(foreground_frame_object_arr_num_frame，foreground_frame_object_arr_height， foreground_frame_object_arr_width， 4)
    foreground_frame_arr = background_frame_object_arr
    for se_index in range(len(selected_objects_arr)):
        if selected_objects_arr[se_index] == 1:
                for h_index in range(foreground_frame_object_arr_height):
                    for w_index in range(foreground_frame_object_arr_width):
                        if foreground_frame_object_arr[se_index][h_index][w_index][3] == 255:
                            foreground_frame_arr[h_index][w_index][0] = foreground_frame_object_arr[se_index][h_index][w_index][0]
                            foreground_frame_arr[h_index][w_index][1] = foreground_frame_object_arr[se_index][h_index][w_index][1]
                            foreground_frame_arr[h_index][w_index][2] = foreground_frame_object_arr[se_index][h_index][w_index][2]
                            foreground_frame_arr[h_index][w_index][3] = 255
    return foreground_frame_arr

def diff_f_and_b(foreground_path, background_path):
    foreground_pic = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
    background_pic = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
    video_height = len(foreground_pic)
    video_width = len(foreground_pic[0])
    for h_index in range(video_height):
        for w_index in range(video_width):
            if foreground_pic[h_index][w_index][3] == 255\
            or h_index == 0 or w_index == 0 or w_index == video_width-1 or h_index==video_height-1\
            or (abs(int(foreground_pic[h_index][w_index][0]) - int(background_pic[h_index][w_index][0]))<10 \
            and abs(int(foreground_pic[h_index][w_index][1]) - int(background_pic[h_index][w_index][1]))<10\
            and abs(int(foreground_pic[h_index][w_index][2]) - int(background_pic[h_index][w_index][2]))<10):
                foreground_pic[h_index][w_index][3] = 0
            else:
                foreground_pic[h_index][w_index][3] = 255
    ker = np.ones((10,10),np.uint8)
    ori_foreground_pic = foreground_pic
    foreground_pic = cv2.morphologyEx(foreground_pic, cv2.MORPH_OPEN,  ker, 1)
    #foreground_pic = cv2.erode(foreground_pic,  ker, 1)
    for h_index in range(video_height):
            for w_index in range(video_width):
                 if foreground_pic[h_index][w_index][3] == 0:
                    ori_foreground_pic[h_index][w_index][0] = 255
                    ori_foreground_pic[h_index][w_index][1] = 255
                    ori_foreground_pic[h_index][w_index][2] = 255
                    ori_foreground_pic[h_index][w_index][3] = 0
                 else:
                    ori_foreground_pic[h_index][w_index][3] = 255

    #cv2.imwrite('../dd/diff.png', ori_foreground_pic)
    return ori_foreground_pic

def find_label_object(foreground_pic, labels):
    for y in range (len(foreground_pic)):
        for x in range (len(foreground_pic[0])):
            if(foreground_pic[y][x][3] == 255):
                count = get_zero_label(labels)
                ret = dfs_label_object(foreground_pic, y, x, count)
                if ret:
                    labels[count] = 1


def dfs_label_object(foreground_pic, index_y,index_x, label):
    is_exist = False
    dirs = [[1,0],[0,1],[0,-1],[-1,0],[1,1],[-1,1],[1,-1],[-1,-1]]
    stack = collections.deque()
    stack.append([index_y,index_x])
    while stack:
        y,x= stack.popleft()
        if(y<0 or x<0 or y >= len(foreground_pic) or x>=len(foreground_pic[0]) or foreground_pic[y][x][3] != 255):
            continue
        foreground_pic[y][x][3] = label
        is_exist = True
        for dir in dirs:
           stack.append([y+dir[0],x+dir[1]])
    return is_exist

def find_objects_on_page(prev_pic, cur_pic, labels):

    for y in range (len(cur_pic)):
            for x in range (len(cur_pic[0])):
                if(cur_pic[y][x][3]==255 and prev_pic[y][x][3] != 0 and prev_pic[y][x][3] != 255):
                    label = prev_pic[y][x][3]
                    ret =  dfs_label_object(cur_pic,y,x,label)
                    if ret:
                        labels[label]+=1

    find_label_object(cur_pic, labels)

def get_zero_label(labels):
    for i in range(1,len(labels)):
        if labels[i] == 0:
            return i
    return -1
def change_label_to_255(cur_pic, label):
    new_pic = cur_pic.copy()
    for y in range (len(cur_pic)):
        for x in range (len(cur_pic[0])):
             if cur_pic[y][x][3] == label:
                new_pic[y][x][3] = 255
             else:
                new_pic[y][x][3] = 0
    return new_pic

def app_three_remove_object(foreground_p, background_p, frames_number,  selected_objects_arr):
    video_arr = []
    for i in range(frames_number):
        background_path = background_p + "noobjectvideo_" + str(i) + ".png"
        foreground_frame_arr = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)
        foreground_frame_object_arr = []
        for j in range(len(selected_objects_arr)):
            if(selected_objects_arr[j] == 1):
                fore_path = foreground_p + "new_fore_" + str(j) + "_"+str(i) + ".png"
                fore_read = cv2.imread(fore_path, cv2.IMREAD_UNCHANGED)
                foreground_frame_object_arr.append(fore_read)
        foreground_frame_object_arr_height = len(foreground_frame_object_arr[0])
        foreground_frame_object_arr_width = len(foreground_frame_object_arr[0][0])
        #foreground_frame_arr = background_frame_object_arr
        for se_index in range(len(foreground_frame_object_arr)):
                    for h_index in range(foreground_frame_object_arr_height):
                        for w_index in range(foreground_frame_object_arr_width):
                            if foreground_frame_object_arr[se_index][h_index][w_index][3] == 255:
                                foreground_frame_arr[h_index][w_index][0] = foreground_frame_object_arr[se_index][h_index][w_index][0]
                                foreground_frame_arr[h_index][w_index][1] = foreground_frame_object_arr[se_index][h_index][w_index][1]
                                foreground_frame_arr[h_index][w_index][2] = foreground_frame_object_arr[se_index][h_index][w_index][2]
                                foreground_frame_arr[h_index][w_index][3] = 255
        video_arr.append(foreground_frame_arr)
        convert_bgra_2_video(video_arr,30)

def transfer(foreground_p, background_p,foreground_list, saveing_path):
    new_foreground_pic = []
    label = 0
    labels = np.zeros(255)
    prev = labels.copy()
    for i in range(len(foreground_list)):
        foreground_path = foreground_p + foreground_list[i]
        background_path = background_p + "noobjectvideo_" + str(i) + ".png"
        foreground_pict = diff_f_and_b(foreground_path, background_path)
        if(i == 0):
            find_label_object(foreground_pict, labels)
        else:
            find_objects_on_page(new_foreground_pic[i-1], foreground_pict, labels)
        new_foreground_pic.append(foreground_pict)
        for j in range(len(labels)):
            if 0 < labels[j] < 10 and labels[j] == prev[j]:
                labels[j] = 0
        prev = labels.copy()
        #cv2.imwrite(saveing_path+"new_fore"+str(i)+".png", foreground_pict)


    for i in range(1,len(labels)):
        if labels[i] >= 10:
            print(i)
            for j in range(len(foreground_list)):
                new_pic = change_label_to_255(new_foreground_pic[j], i)
                cv2.imwrite(saveing_path+"new_fore_"+str(i)+"_"+str(j)+".png", new_pic)

if __name__ == "__main__":
    #video_path = "/Users/zihao/Documents/GitHub/remake-video-project/video_view/video2.mp4"
    #video_path = "C:\\Users\\14048\\Desktop\\multimedia\\project\\video_view/video2.mp4"
    #arr,fps = convert_video_2_bgra(video_path)
    #convert_bgra_2_video(arr, fps)
    res_path = "/Users/zihao/Desktop/USC/576/res"
    foreground_num = 50
    #read_file_of_res_run_app_one(res_path,foreground_num)
    '''
    fps = 30
    height = 448
    width = 896
    path = []
    test_mode = True
    read_file_of_res_run_app_two(res_path, foreground_num, fps,  height, width, path, test_mode)
    '''
    foreground_p = "/Users/zihao/Desktop/USC/576/data/"
    foreground_list = []
    for file in os.listdir(foreground_p):
        if file.endswith(".png"):
            foreground_list.append(file)

    foreground_list.sort()
    background_p= "/Users/zihao/Desktop/USC/576/noobjectvideo/"
    saveing_path = "/Users/zihao/Desktop/USC/576/dd/"
    #find object number
    transfer(foreground_p,background_p,foreground_list, saveing_path)

    #background_list = os.listdir(background_p)
    #background_list.sort()
    '''
    frames_number = len(foreground_list)
    selected_objects_arr = [0,1]
    app_three_remove_object(saveing_path, background_p, frames_number, selected_objects_arr)
    '''

