#import cv2 package
import cv2
import cv_bridge
from cv_bridge.boost.cv_bridge_boost import getCvType

#import ros package
import rospy
from sensor_msgs.msg import Image

bridge = cv_bridge.CvBridge()
FILE_PATH = '/home/lai/Desktop/dataset/Tree2/'
IMAGE_HEAD = 'tree2_'
IMAGE_FORMAT = '.jpg'
IMAGE_ID = 0

class camera_capture:
    def rgb_cb(self, data):
        self.rgb_image = self.bridge.imgmsg_to_cv2(data)
        cv2.imshow("rgb camera",self.rgb_image)
        global IMAGE_ID
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.IMG_SAVE = FILE_PATH + IMAGE_HEAD + str(IMAGE_ID) + IMAGE_FORMAT
            self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.IMG_SAVE,self.rgb_image)
            IMAGE_ID += 1
            print("captured and saved!")
        print("callback finish!")

    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        #use $rostopic list to find out the correct topic about images
        self.image_sub = rospy.Subscriber('/d435/color/image_raw', Image, self.rgb_cb)

rospy.init_node('camera_capture')
main = camera_capture()
rospy.spin()
# END ALL