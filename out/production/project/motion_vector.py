import numpy as np
import cv2
import random
import time
import os
import sys
import math
import collections

def YCrCb2BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

def BGR2YCrCb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def get_blockized_height_width(frame, block_size):
    # frame need to be 2d
    # cut out edge here
    h, w = frame.shape
    h_block_num = int(h / block_size)
    w_block_num = int(w / block_size)
    return h_block_num * block_size, w_block_num * block_size


def preprocess_a_frame(frame_path_or_ndarray, block_size):
    # get luminosity and blockize

    if isinstance(frame_path_or_ndarray, str):
        frame_Y = BGR2YCrCb(cv2.imread(frame_path_or_ndarray))[:, :, 0] # get luminosity

    elif isinstance(frame_path_or_ndarray, np.ndarray):
        frame_Y = BGR2YCrCb(frame_path_or_ndarray)[:, :, 0] # get luminosity

    else:
        raise ValueError

    #resize frame to fit segmentation
    blockized_h, blockized_w = get_blockized_height_width(frame_Y, block_size)
    frame_Y_blockized = cv2.resize(frame_Y, (blockized_w, blockized_h))

    return frame_Y_blockized

def getBlockZone(coord, search_area, block, block_size):
    px, py = coord # coordinates of macroblock center
    px, py = px-int(block_size/2), py-int(block_size/2) # get top left corner of macroblock
    px, py = max(0,px), max(0,py) # ensure macroblock is within bounds

    block_to_compare = search_area[py:py+block_size, px:px+block_size] # retrive macroblock from anchor search area

    try:
        assert block_to_compare.shape == block.shape # must be same shape
    except Exception as e:
        print(e)
        print(f"ERROR - ABLOCK SHAPE: {block_to_compare.shape} != TBLOCK SHAPE: {block.shape}")

    return block_to_compare

def blockwise_MAD(block_a, block_b):
    return np.sum(np.abs(np.subtract(block_a, block_b)))/(block_a.shape[0]*block_a.shape[1])

def TSS(block, search_area, block_size, search_expand_length): #3 Step Search
    # Three Step Searching
    # MAD
    step = 4
    ah, aw = search_area.shape
    acy, acx = int(ah/2), int(aw/2) # get center search area

    minMAD = float("+inf")
    minP = None

    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx+step, acy)
        p3 = (acx, acy+step)
        p4 = (acx+step, acy+step)
        p5 = (acx-step, acy)
        p6 = (acx, acy-step)
        p7 = (acx-step, acy-step)
        p8 = (acx+step, acy-step)
        p9 = (acx-step, acy+step)
        coords = [p1,p2,p3,p4,p5,p6,p7,p8,p9] # retrieve 9 search points

        for coord in coords:
            block_to_compare = getBlockZone(coord, search_area, block, block_size) # get anchor macroblock, coord serve as center
            MAD = blockwise_MAD(block, block_to_compare) # determine MAD
            if MAD < minMAD: # store point with minimum mAD
                minMAD = MAD
                minP = coord

        step = int(step/2)

    px_center, py_center = minP # center of matched block
    px_top_left, py_top_left = px_center - int(block_size / 2), py_center - int(block_size / 2) # get top left corner of minP
    px_top_left, py_top_left = max(0, px_top_left), max(0, py_top_left) # ensure minP is within bounds
    matchBlock = search_area[py_top_left:py_top_left + block_size, px_top_left:px_top_left + block_size] # retrieve best macroblock from anchor search area

    return matchBlock, px_top_left - search_expand_length, py_top_left - search_expand_length

def hierarchical_search(base_block, area_to_searched, block_size, search_expand_length, max_best_candidates_per_level = 1, best_candidates_SAD_no_bigger_than_minSAD_tolerate_range = 500):
    # modified
    #                                                      n must be 2^x            k                                              if set tolerate range, need to set for each level
    # the return is based on frame_being_searched left_top_corner, matrix is [[dx0,dy0][dx1,dy1]...] [candidate idx][0:dy 1:x]
    def maintain_candidates(SAD, candidates, candidates_SADs, candidates_SAD_max, candidates_SAD_min, y_of_candidate, x_of_candidate):
        if candidates_SAD_max[0] == 0:
            # cache is filled with SAE=0 candidates, do not insert
            return
        if max_best_candidates_per_level > 1:
            if SAD < candidates_SAD_min[0] + best_candidates_SAD_no_bigger_than_minSAD_tolerate_range:
                if len(candidates) == max_best_candidates_per_level:
                    index_to_pop = candidates_SADs.index(candidates_SAD_max[0])
                    candidates.pop(index_to_pop)
                    candidates_SADs.pop(index_to_pop)
                candidates.append((y_of_candidate, x_of_candidate))
                candidates_SADs.append(SAD)
                if candidates_SAD_max[0] == math.inf:
                    candidates_SAD_max[0] = 0
                candidates_SAD_max[0] = max(candidates_SADs)
                if SAD < candidates_SAD_min[0]:
                    candidates_SAD_min[0] = SAD
                    for i in range(len(candidates)-1, -1, -1):
                        if candidates_SADs[i] > candidates_SAD_min[0] + best_candidates_SAD_no_bigger_than_minSAD_tolerate_range:
                            # print("pop", candidates_SADs[i], "because of out of range", candidates_SAD_min[0] + best_candidates_SAD_no_bigger_than_minSAD_tolerate_range)
                            candidates.pop(i)
                            candidates_SADs.pop(i)
        elif max_best_candidates_per_level == 1:
            if len(candidates_SADs) == 0:
                candidates_SADs.append(SAD)
                candidates.append((y_of_candidate, x_of_candidate))
            if SAD < candidates_SADs[0]:
                candidates_SADs[0] = SAD
                candidates[0] = (y_of_candidate, x_of_candidate)
            # print(SAD, candidates_SADs, candidates_SAD_min, candidates_SAD_max)


    frame_height, frame_width = area_to_searched.shape
    # print(frame_height, frame_width)
    # if frame_height % block_size != 0 or frame_width % block_size != 0:
    #     raise Exception("search_expand_length is not multiple of block size")
    level_num = 4
    n = int(block_size/(math.pow(2, level_num-1)))
    k = int(search_expand_length/(math.pow(2, level_num-1)))
    # min sum absulute diff
    min_SAD = math.inf
    candidates = [] # to be faster could use heapq, maxpq, pop max when too much
    candidates_SADs = []
    candidates_SAD_max = [math.inf]
    candidates_SAD_min = [math.inf]
    # full search at level highest
    # everything is left_corner
    # level 4 step 4
    step = int(math.pow(2, level_num-1))
    for y_of_candidate in range(0, frame_height-block_size+1, step):
        for x_of_candidate in range(0, frame_width-block_size+1, step):
            # print(y_of_candidate, x_of_candidate)
            # blockwise
            SAD = int(0)
            # count1 = 0 # just check
            for relative_y in range(0, block_size, step):
                for relative_x in range(0, block_size, step):
                    # count1 += 1
                    y_in_frame = y_of_candidate + relative_y
                    x_in_frame = x_of_candidate + relative_x
                    SAD += abs(int(base_block[relative_y][relative_x]) - int(area_to_searched[y_in_frame][x_in_frame]))
            maintain_candidates(SAD, candidates, candidates_SADs, candidates_SAD_max, candidates_SAD_min, y_of_candidate, x_of_candidate)
    # print(candidates, candidates_SADs)
    
    for level in range(level_num-1, 0, -1):
        # level 3 step 2^2
        # level 2 step 2^1
        # level 1 step 2^0
        last_iteration_candidates = candidates

        step = int(math.pow(2, level-1))
        min_SAD = math.inf
        candidates = [] # to be faster could use heapq, maxpq, pop max when too much
        candidates_SADs = []
        candidates_SAD_max = [math.inf]  # for referential change, access as arr[0]
        candidates_SAD_min = [math.inf]  # for referential change

        visited = set()
        for last_iteration_candidate in last_iteration_candidates:
            # print(last_iteration_candidate)

            # -1 0 1
            for y_of_candidate in range(max(0, last_iteration_candidate[0]-step), min(frame_height-block_size+1, last_iteration_candidate[0]+step*2), step):
                for x_of_candidate in range(max(0, last_iteration_candidate[1]-step), min(frame_width-block_size+1, last_iteration_candidate[1]+step*2), step):
                    candidate = (y_of_candidate, x_of_candidate)
                    if candidate in visited:
                        continue
                    visited.add(candidate)
                    # if y_of_candidate == last_iteration_candidate[0] and x_of_candidate == last_iteration_candidate[1]:
                    #     continue
                    # blockwise
                    SAD = int(0)
                    # count1 = 0 # just check
                    for relative_y in range(0, block_size, step):
                        for relative_x in range(0, block_size, step):
                            # count1 += 1
                            y_in_frame = y_of_candidate + relative_y
                            x_in_frame = x_of_candidate + relative_x
                            SAD += abs(int(base_block[relative_y][relative_x]) - int(area_to_searched[y_in_frame][x_in_frame]))
                    # print(candidate, SAD)
                    maintain_candidates(SAD, candidates, candidates_SADs, candidates_SAD_max, candidates_SAD_min, y_of_candidate, x_of_candidate)
        # print(candidates, candidates_SADs)

    # # at last, check itself
    # y_of_candidate, x_of_candidate = search_expand_length, search_expand_length
    # candidate = (y_of_candidate, x_of_candidate)
    # if candidate not in visited:
    #     SAD = int(0)
    #     # count1 = 0 # just check
    #     for relative_y in range(0, block_size, step):
    #         for relative_x in range(0, block_size, step):
    #             # count1 += 1
    #             y_in_frame = y_of_candidate + relative_y
    #             x_in_frame = x_of_candidate + relative_x
    #             SAD += abs(int(base_block[relative_y][relative_x]) - int(area_to_searched[y_in_frame][x_in_frame]))
    #     maintain_candidates(SAD, candidates, candidates_SADs, candidates_SAD_max, candidates_SAD_min, y_of_candidate, x_of_candidate)
    # # print(candidates, candidates_SADs)

    res = []
    # return a list of coordinates who has the smallest SAD
    candidates_SAD_min[0] = min(candidates_SADs)
    for i in range(len(candidates)):
        if candidates_SADs[i] == candidates_SAD_min[0]:
            res.append(candidates[i])
    # print(res, candidates_SAD_min[0])
    return res, candidates_SAD_min[0]


def residual(target, predicted):
    return np.subtract(target, predicted)

def get_search_area(x, y, frame, block_size, search_expand_length):
    h, w = frame.shape
    # cx, cy = getCenter(x, y, block_size)
    # sx = max(0, cx-int(block_size/2)-searchArea) # ensure search area is in bounds
    # sy = max(0, cy-int(block_size/2)-searchArea) # and get top left corner of search area

    sx = max(0, x-search_expand_length) # ensure search area is in bounds
    sy = max(0, y-search_expand_length) # and get top left corner of search area
    search_area = frame[sy:min(y+search_expand_length+block_size, h), sx:min(x+search_expand_length+block_size, w)]
    return search_area, [sy, min(y+search_expand_length+block_size, h), sx, min(x+search_expand_length+block_size, w)]

def get_motion_vector_matrix(frame_being_searched, frame_base, block_size, search_expand_length=16, if_generate_predict_frames = False):
    #                            frame n             frame n+1                search_expand_length must be multiple of block_size
    # search  frame_base n+1       in     frame_being_searched n
    h, w = frame_being_searched.shape
    predicted = None
    if if_generate_predict_frames:
        predicted = np.ones((h, w))*255
    blockized_h, blockized_w = get_blockized_height_width(frame_being_searched, block_size)
    bcount = 0
    matrix = [[None for i in range(w//block_size)] for j in range(h//block_size)]

    max_best_candidates_per_level = 10

    # for each block
    for y in range(0, blockized_h, block_size):
        for x in range(0, blockized_w, block_size):
            bcount += 1
            base_block = frame_base[y:y+block_size, x:x+block_size]
            search_area, indices = get_search_area(x, y, frame_being_searched, block_size, search_expand_length)
            # print(cur_block.shape, search_area.shape)
            # matchBlock, dx, dy = TSS(cur_block, search_area, block_size, search_expand_length)
            # print(dx, dy)
            
            # best_matches is not the motion vector, is based on clipped search area
            best_matches, SAD = hierarchical_search(base_block, search_area, block_size, search_expand_length, max_best_candidates_per_level)

            # print(SAD//(block_size*block_size)) # this is MAD

            # # multiple candidates usually adjacent, take average
            # if len(best_matches) > 1:
            #     # print("multiple best candidates", SAD, best_matches)
            #     temp0 = 0
            #     temp1 = 0
            #     templ = len(best_matches)
            #     for i in range(templ):
            #         temp0 += best_matches[i][0]
            #         temp1 += best_matches[i][1]
            #     avg_coord1 = temp0//templ
            #     avg_coord2 = temp1//templ
            #     best_matches[0] = (avg_coord1, avg_coord2)
    

            motion_vectors = []
            for best_match in best_matches:
                match_relative_y, match_relative_x = best_match
                match_y = match_relative_y + max(0, y-search_expand_length) # match position in n
                match_x = match_relative_x + max(0, x-search_expand_length)
                dx = x - match_x
                dy = y - match_y
                # if dx != 0 or dy != 0:
                #     print(y,x, match_y,match_x,dy,dx,SAD, indices, search_area.shape)
                motion_vectors.append([dy , dx])

            # motion points from n to n+1
            # [y][x][candidate idx][0:dy 1:x]
            matrix[y//block_size][x//block_size] = motion_vectors
            if if_generate_predict_frames:
                predicted[y:y+block_size, x:x+block_size] = frame_being_searched[match_y:match_y+block_size, match_x:match_x+block_size]

    # assert bcount == int(blockized_h / block_size * blockized_w / block_size) #check all macroblocks are accounted for
    return matrix, predicted

def save_intermediate_images(frame_base, predicted, intermediate_output_dict="OUTPUT", idx = None):
    residualFrame = residual(frame_base, predicted)

    
    isdir = os.path.isdir(intermediate_output_dict)
    if not isdir:
        os.mkdir(intermediate_output_dict)


    if idx:
        # cv2.imwrite(f"{outfile}/targetFrame.png", targetFrame)
        cv2.imwrite(f"{intermediate_output_dict}/predictedFrame"+"{:03d}".format(idx)+".png", predicted)
        cv2.imwrite(f"{intermediate_output_dict}/residualFrame"+"{:03d}".format(idx)+".png", residualFrame)
        # cv2.imwrite(f"{outfile}/reconstructTargetFrame.png", reconstructTargetFrame)
        # cv2.imwrite(f"{outfile}/naiveResidualFrame.png", naiveResidualFrame)
        # resultsFile = open(f"{outfile}/results.txt", "w"); resultsFile.write(f"{rmText}\n{nrmText}\n"); resultsFile.close()

            #print("AnchorSearchArea: ", anchorSearchArea.shape)

            # anchorBlock, px, py = getBestMatch(cur_block, search_area, block_size) #get best anchor macroblock
    else:
        
        cv2.imwrite(f"{intermediate_output_dict}/predictedFrame.png", predicted)
        cv2.imwrite(f"{intermediate_output_dict}/residualFrame.png", residualFrame)

def draw_line_on_predicted(predicted, motion_vector_matrix, block_size):
    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    h, w = predicted.shape
    for vector_y in range(len(motion_vector_matrix)):
        for vector_x in range(len(motion_vector_matrix[0])):
            for motion_vector_matrix_pair in motion_vector_matrix[vector_y][vector_x]:
                dy, dx = motion_vector_matrix_pair
                # x01y01just for calculate (x0, y0) (x1, y1)
                y0 = block_y = vector_y*block_size + block_size//2 # center
                x0 = block_x = vector_x*block_size + block_size//2
                y1 = last_block_y = block_y - dy
                x1 = last_block_x = block_x - dx
                steep = abs(dy) > abs(dx)
                if steep:
                    x0, y0 = y0, x0
                    x1, y1 = y1, x1
                if x0>x1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0
                deltax = x1-x0
                deltay = abs(y1-y0)
                error = deltax//2
                y = y0
                ystep = 1 if y0<y1 else -1
                for x in range(x0, x1):
                    if steep:
                        #(y,x) is position following (x, y) format
                        # predicted following (y, x) format
                        if 0<=y<h and 0 <=x<w:
                            predicted[x][y] = 255
                    else:#(x,y)
                        if 0<=y<h and 0 <=x<w:
                            predicted[y][x] = 255
                    error -= deltay
                    if error < 0:
                        y += ystep
                        error += deltax