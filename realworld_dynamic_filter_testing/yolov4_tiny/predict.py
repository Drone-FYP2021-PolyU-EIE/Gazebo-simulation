import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #-------------------------------------------------------------------------#
    #   mode means the detection mode
    #   'predict' for single picture detection
    #   'video' for video detection
    #   'fps' for testing fps
    #-------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   video_path means the video path if video_path=0 means the camera
    #   video_save_path means the detection video result path if video_save_path nothing means not save
    #   video_fps means save the video fps
    #   video_path、video_save_path and video_fps only work when the mode='video'
    #   save the video required to use ctrl+c otherwise it cannot save the detection video
    #-------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                left, right = yolo.detection_result()
                print ("Turn left : {}".format(left))
                print ("Turn right : {}".format(right))
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            # Get frame
            ref,frame=capture.read()
            # BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # convet to Image
            frame = Image.fromarray(np.uint8(frame))
            # perform detection
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR to fulfil the opencv format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        test_interval = 100
        img = Image.open('/home/lai/Desktop/custom_deep_learning/yolov3/test_image.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")
