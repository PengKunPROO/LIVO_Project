#pragma once
#include "var_global.h"
#include "common_lib.h"

class Frame
{
private:
    PointCloudXYZI::Ptr m_frame_points;
    //world -> frame transformation
    Eigen::Isometry3d m_pose;
    //img corresponding to m_frame_points
    cv::Mat m_img;
    int id;
    bool isFEOptFinished;
    bool isPublished;
public:
    Frame(/* args */);
    Frame(PointCloudXYZI::Ptr frame_points,Eigen::Isometry3d pose,cv::Mat img);
    Frame(const Frame& copy);
    ~Frame();

public:
    static int uID;

/* Access Function*/
public:
    Eigen::Isometry3d pose(){
        return m_pose;
    }
    void setPose(Eigen::Isometry3d modify){
        m_pose = modify;
    }
    cv::Mat image(){
        //这里是引用
        return m_img;
    }
    PointCloudXYZI::Ptr points_map(){
        //一样，都是shared ptr，所以要注意使用
        return m_frame_points;
    }
    int ID(){return id;}
    void setFrameOptStatus(bool isFinished){
        isFEOptFinished = isFinished;
    }
    void setPublishedStatus(bool isPub){
        isPublished = isPub;
    }
    bool OptStatus(){return isFEOptFinished;}
    bool published(){return isPublished;}
};


