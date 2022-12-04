import numpy as np
import cv2

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
        if(cv2.waitKey(1)==ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

    return (arr,fps)

def convert_bgra_2_video(arr, fps):

    height, width = len(arr[0]), len(arr[0][0])
    out = cv2.VideoWriter("new_video1.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    for i in range(len(arr)):
        out.write(cv2.cvtColor(arr[i], cv2.COLOR_BGRA2BGR))
    out.release()

def app_one_display_trails(foreground_frame_arr, foreground_element_num, background_panorama):

    display_trails_pic = background_panorama
    display_trails_pic_height = len(display_trails_pic[0])
    display_trails_pic_width = len(display_trails_pic[0][0])
    interval_between_frames = (int)(len(foreground_frame_arr)/foreground_element_num)
    for i in range(foreground_element_num):
        for h_index in range(display_trails_pic_height):
            for w_index in range(display_trails_pic_width):
                #load alpha data of foreground_frame_arr. if it is 255 replace it with the display_trails_pic.
                if foreground_frame_arr[i*interval_between_frames][h_index][w_index][3] == 255:
                    display_trails_pic[h_index][w_index][0] = foreground_frame_arr[i*interval_between_frames][h_index][w_index][0]
                    display_trails_pic[h_index][w_index][1] = foreground_frame_arr[i*interval_between_frames][h_index][w_index][1]
                    display_trails_pic[h_index][w_index][2] = foreground_frame_arr[i*interval_between_frames][h_index][w_index][2]
    save_pic_status = cv2.imwrite('app_one.png', display_trails_pic)
    print("App one Image written save file: ", save_pic_status)
#location_path is a n*3 array which [y, x, time] foreground_frame_arr have all foreground
def app_two_create_video(foreground_frame_arr, background_panorama, fps, video_height, video_width, location_path):
    frame_arr = []
    start_frame_location = location_path[0]
    for path_index in range(1,len(location_path)):
        next_frame_location = location_path[path_index]
        number_of_frame_between_interval = (next_frame_location[2] - start_frame_location[2]) * fps
        height_change = (next_frame_location[0] - start_frame_location[0])/number_of_frame_between_interval
        height_change = (int) (height)
        width_change = (next_frame_location[1] - start_frame_location[1])/number_of_frame_between_interval
        width_change = (int) (width)
        for frame_index in range(number_of_frame_between_interval):
            display_pic = [[[0,0,0,0]*width]*height]
            for h_index in range(video_height):
                for w_index in range(video_width):
                    if foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][3] == 255:
                        display_pic[h_index][w_index][0] = foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][0]
                        display_pic[h_index][w_index][1] = foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][1]
                        display_pic[h_index][w_index][2] = foreground_frame_arr[start_frame_location[2]+frame_index][start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][2]
                        display_pic[h_index][w_index][3] = 255
                    else:
                        display_pic[h_index][w_index][0] = background_panorama[start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][0]
                        display_pic[h_index][w_index][1] = background_panorama[start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][1]
                        display_pic[h_index][w_index][2] = background_panorama[start_frame_location[0]+frame_index+h_index][start_frame_location[1]+frame_index+w_index][2]
                        display_pic[h_index][w_index][3] = 255
            frame_arr.append(display_pic)
        start_frame_location = next_frame_location
    convert_bgra_2_video(frame_arr, fps)

def app_three_remove_object():
       print("..")


if __name__ == "__main__":
    video_path = "/Users/zihao/Documents/GitHub/remake-video-project/video_view/video2.mp4"
    #video_path = "C:\\Users\\14048\\Desktop\\multimedia\\project\\video_view/video2.mp4"
    arr,fps = convert_video_2_bgra(video_path)
    print(fps)
    #convert_bgra_2_video(arr, fps)
