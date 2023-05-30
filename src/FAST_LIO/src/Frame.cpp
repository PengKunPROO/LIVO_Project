#include"Frame.h"

//每个类对象拥有唯一ID，不停累加
int Frame::uID=0;

Frame::Frame(/* args */)
{
}

Frame::Frame(PointCloudXYZI::Ptr frame_points,Eigen::Isometry3d pose,cv::Mat img){
    //因为是共享指针，所以相当于这里两个虽然指向同一个内存，但是其指针本身的地址不一样
    m_frame_points = frame_points;
    //opencv也一样，构造函数给的是引用，所以传进来要先clone，这样m_img就成了唯一指向该地址的指针了，当构造结束和主线程临时变量析构后
    m_img = img;
    m_pose = pose;
    id = uID;
    uID++;
    isFEOptFinished = false;
    isPublished = false;
}

//深度复制的拷贝构造函数，都是开辟自己的内存
Frame::Frame(const Frame& copy){
    PointCloudXYZI::Ptr temp(new PointCloudXYZI());
    pcl::copyPointCloud(*(copy.m_frame_points),*temp);
    m_frame_points = temp;
    m_img = copy.m_img.clone();
    m_pose = copy.m_pose;
    id = copy.id;
    isFEOptFinished = copy.isFEOptFinished;
    isPublished = copy.isPublished;
}

Frame::~Frame()
{
    //虽然感觉没啥用，这部分应该会自己释放，但是还是手动调用下，安全起见
    m_img.release();
    m_frame_points->clear();
    m_frame_points.reset();
}