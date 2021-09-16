import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class test():
    def __init__(self):
        rospy.init_node("test")
        # Subscribe camera info [depth_rgb aligned]
        self.marker_vis = rospy.Publisher("/d435/realsense_image_marker",Marker,queue_size=1)

    # Create marker in rviz
    def create_marker(self,x1,y1,z1,r,g,b,id):
        self.marker = Marker()
        self.marker.header.frame_id = 'world'
        self.marker.header.stamp = rospy.Time.now()
        self.marker.ns = 'realsense_marker'
        self.marker.action = self.marker.ADD
        self.marker.id = id
        self.marker.type = Marker.SPHERE_LIST
        self.marker.pose.orientation.w = 1.0
        self.start = Point(x1,y1,z1)
        self.marker.points.append(self.start)
        self.marker.scale.x = 0.05
        self.marker.scale.z = 0.05
        self.marker.scale.y = 0.05
        self.marker.color.a = 1.0
        self.marker.color.r = r
        self.marker.color.g = g
        self.marker.color.b = b
        self.marker_vis.publish(self.marker)
    
if __name__ == '__main__':
    # Delay for tf capture
    print("Start")
    while not rospy.is_shutdown():
        hi = test()
        #hi.create_marker(-0.053951529,0.751,0.161854587,1,0,1,id=0)
        hi.create_marker(-0.474141269,0.750,0.096983441,1,0,1,id=0)
        print("ok")