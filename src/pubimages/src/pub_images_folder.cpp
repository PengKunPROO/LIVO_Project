#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <boost/filesystem.hpp>
#include <string>

using namespace boost::filesystem;

std::vector<std::string> GetFileName(std::string dir){

    std::vector<std::string> files;
    for (auto i = directory_iterator(dir); i != directory_iterator(); i++){
        if (!is_directory(i->path())) //we eliminate directories in a list
        {
            files.push_back(i->path().filename().string());
        }
        else
            continue;
    }
    return files;
}

bool AscendingSort(std::string x,std::string y){
    if(std::stod(x)==std::stod(y)) {
        return false;
    }
    else {
        return std::stod(x)<std::stod(y);
    }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("camera/image", 10000);
  std::vector<std::string> all = GetFileName("/opt/cache");
  for(int i=0;i<all.size();i++){
    int end = all[i].size()-1;
    all[i] = all[i].substr(0,end-3);
    ROS_INFO("%s",all[i].c_str());
  }
    ROS_INFO("-----------------------------------------------");
 
    std::vector<double> temp;
    //   for(int i=0;i<all.size();i++){
    //     temp.push_back(std::stod(all[i]));
    //     }
   std::sort(all.begin(),all.end(),AscendingSort);
   std::cout.precision(18);
   for(int i=0;i<all.size();i++){
        std::cout<<"pic name: "<<all[i]<<std::endl;
    }
    ros::Duration(2).sleep();
    ros::Rate loop_rate(30);
    int i=0;
    std_msgs::Header header;
    while (ros::ok()) {
        if(i>all.size()-1){
            ROS_INFO("All the pics have been sent sunccessfully!");
            break;
        }
    std::string img_name = "/opt/cache/"+all[i]+".jpg";
    cv::Mat image = cv::imread(img_name);
    if(image.empty()){
      ROS_INFO("open error\n %s", img_name.c_str());
    }
    //图片信息头的id就是自己的名字，也就是对应的时间戳，不过这个时间戳是模拟出来的，我觉得是不准的，但是目前只能这样做
    header.frame_id=all[i];
    header.stamp=ros::Time::now();
    header.seq = i;
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    pub.publish(msg);
    ROS_INFO("send the image %s successfully!",all[i].c_str());
    i++;
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}