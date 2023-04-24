#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

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


int main(void){
    std::vector<std::string> all = GetFileName("/opt/Camera-Lidar-Data/frames_10");

    std::vector<double> intrinsic = {817.846,0,1000.37,0,835.075,551.246,0,0,1};
    std::vector<double> distortion = {-0.2658,0.0630,0.0026,0.0015,0};
    std::vector<double> extrinsic = {-0.00589083,-0.99976,-0.0211195,0.0221539,-0.0232254,0.021251,-0.999504,0.0745742,0.999713,-0.0053974,-0.023345,0.675224,0,0,0,1};
    double matrix1[3][3] = {{intrinsic[0], intrinsic[1], intrinsic[2]}, {intrinsic[3], intrinsic[4], intrinsic[5]}, {intrinsic[6], intrinsic[7], intrinsic[8]}}; 
    double matrix2[3][4] = {{extrinsic[0], extrinsic[1], extrinsic[2], extrinsic[3]}, {extrinsic[4], extrinsic[5], extrinsic[6], extrinsic[7]}, {extrinsic[8], extrinsic[9], extrinsic[10], extrinsic[11]}};
    
    cv::Mat matrix_in = cv::Mat(3, 3, CV_64F, matrix1);
    cv::Mat matrix_out = cv::Mat(3, 4, CV_64F, matrix2);

    // set intrinsic parameters of the camera
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = intrinsic[0];
    camera_matrix.at<double>(0, 2) = intrinsic[2];
    camera_matrix.at<double>(1, 1) = intrinsic[4];
    camera_matrix.at<double>(1, 2) = intrinsic[5];

	// set radial distortion and tangential distortion
    cv::Mat distortion_coef = cv::Mat::zeros(5, 1, CV_64F);
    distortion_coef.at<double>(0, 0) = distortion[0];
    distortion_coef.at<double>(1, 0) = distortion[1];
    distortion_coef.at<double>(2, 0) = distortion[2];
    distortion_coef.at<double>(3, 0) = distortion[3];
    distortion_coef.at<double>(4, 0) = distortion[4];

    for(int i=0;i<all.size();i++){
        int end = all[i].size()-1;
        all[i] = all[i].substr(0,end-3);
        ROS_INFO("%s",all[i].c_str());
    }
    int size = all.size();
    ROS_INFO("All we have are %d images!",size);
    for(int i=0;i<size;i++){
        std::string img_name = "/opt/Camera-Lidar-Data/frames_10/"+all[i]+".jpg";
        cv::Mat image = cv::imread(img_name);
        if(image.empty()){
            ROS_INFO("empty error image! %s \n",img_name.c_str());
        }
        cv::Mat view, rview, map1, map2;
        cv::Size imageSize = image.size();
        cv::initUndistortRectifyMap(camera_matrix, distortion_coef, cv::Mat(),cv::getOptimalNewCameraMatrix(camera_matrix, distortion_coef, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
        
        cv::remap(image, image, map1, map2, cv::INTER_NEAREST);  // correct the distortion
        std::string img_name_aft = "/opt/Camera-Lidar-Data/frames_10_undistort/"+all[i]+".jpg";
        cv::imwrite(img_name_aft,image);
        ROS_INFO("Completed! %s",img_name_aft.c_str());
    }


    return 0;
}