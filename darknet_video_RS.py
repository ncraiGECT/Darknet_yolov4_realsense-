
import math
import random
import os
import cv2
import numpy as np
import darknet
import statistics
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_profile =rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
print("Depth Scale is: " , depth_scale)
align_to = rs.stream.color
align = rs.align(align_to)
pc = rs.pointcloud()














def get_object_depth(aligned_depth_frame, bounds):
    area_div = 2
    
    x, y, w, h = bounds[0][0],bounds[0][1],bounds[0][2],bounds[0][3]
    h = math.floor(x*640./608.)
    w = math.floor(y*480./608.)    
    x_vect = []
    y_vect = []
    z_vect = []       
   
    for j in range(h - area_div, h + area_div):
        for i in range(w- area_div, w+ area_div):   
            z = aligned_depth_frame.get_distance(i, j)
            if not np.isnan(z) and not np.isinf(z):
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x,y], z)
                x_vect.append(point[0])
                y_vect.append(point[1])
                z_vect.append(point[2])
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass
    distance= math.sqrt(x_median * x_median + y_median * y_median + z_median * z_median)

    return distance
 


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img,distance):
    x, y, w, h = detections[0][0],detections[0][1],detections[0][2],detections[0][3]
    x = math.floor(x*640./608.)
    y = math.floor(y*480./608.)
    w = math.floor(w*640./608.)
    h = math.floor(h*480./608.)    
    
    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
    cv2.putText(img," person",(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    cv2.putText(img,str(distance),(pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return img


netMain  = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
 
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
 
    
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
 
    print("Starting the YOLO loop...")


    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
		
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) 
        
     
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)                                 
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        LL = len(detections)
        encoding = 'utf-8'
        #print(detections)
  
        for i in range(LL) :
            KK = detections[i][0]
            strr= KK.decode(encoding)
            result = strr.find('person')
            if result is 0:
                detection = detections[i][2:5]
                prob = detections[i][1]
                '''
                if prob <0.8 :
                    continue     
                '''
                distance = get_object_depth(aligned_depth_frame, detection)
                print('Dist=....' ,distance,'probability=....' ,prob)                
                image = cv2.resize(frame_resized,(640,480),
                                   interpolation=cv2.INTER_LINEAR)                
                image = cvDrawBoxes(detection, image,distance)                
               
                cv2.imshow('Demo', image)
                key = cv2.waitKey(3)
                
                
                if key == ord('q'):
                    break
if __name__ == "__main__":
    YOLO()
