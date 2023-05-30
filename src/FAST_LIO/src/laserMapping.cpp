// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>

#include <ikd-Tree/ikd_Tree.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>
#include <thread>
#include <sys/time.h>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/robust_kernel_impl.h"
#include <g2o/types/slam3d/types_slam3d.h>

#include <Eigen/StdVector>
#include <sstream>
#include <ctime>
#include <chrono>

#include "preprocess.h"
#include "pose_optimize.h"
#include "IMU_Processing.hpp"
#include "BackendThread.hpp"
#include "FrontEndThread.hpp"
#include "Frame.h"

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

typedef struct imgStruct
{
    cv::Mat img_data;
    // I will search the nearest img from current timestamp according to the img name
    std::string name;
    double timestamp;
} ImgStore;

// 存的是我们在插值部分计算的每一帧从当前帧转到产生错位的img对应的lidar坐标系的位姿，每一帧都更新
Eigen::Isometry3d lidar_OptLidar;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

/***Opt Log Variables***/
int invalid_count = 0;
Eigen::Isometry3d origTcw;
Eigen::Isometry3d optTcw;
int valid_orig_count = 0;
float before_opt_valid_ratio = 0.0;
float after_opt_valid_ratio = 0.0;
/***********************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;
// TODO: log时间阶段的标志，以后会删
bool first_first_point = true;
bool first_first_point_opt = true;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
// 以程序开始运行的时间作为固定的文件开头
std::string dateString;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, last_timestamp_image = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
double filter_size_camera_opt = 0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0, pcd_index_rgb = 0;
int opt_frame_fn = 0, opt_num_in_one_frame = 0, backend_opt_pose_num = 0;
int rgb_point_save_num = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, imu_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

bool imgChanged = true;
double simulate_time_lidar = 0.0;
double whole_time_one_frame = 0.0f;
double whole_time_one_frame_start = 0.0f;

int row, col;

// 从0开始的frame序号，与前端的优化帧序号对应
int frame_index = 0;
// the num to stop the program
int frame_num_to_stop = 0;
// the num of img we project the point onto
int frame_map_to_save = 0;

int times_pub = 0;

std::vector<vector<int>> pointSearchInd_surf;
std::vector<BoxPointType> cub_needrm;
std::vector<PointVector> Nearest_Points;
std::vector<cv::Point2f> cache_lidar_2_img;

std::vector<LogVariable> cache_all_log;

std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);
std::vector<double> lidar_cam_ext(16, 0.0);
std::vector<double> cam_intrinsic(9, 0.0);
std::deque<double> time_buffer;
std::deque<double> img_time_buffer;
std::deque<PointCloudXYZI::Ptr> lidar_buffer;
std::deque<PointCloudXYZRGB::Ptr> opt_lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<cv::Mat> img_buffer_pck;
std::deque<ImgOpt *> multi_opt_buffer;
std::deque<Frame *> frame_opt_buffer;
std::deque<Frame *> backend_opt_buffer;
// 当缓存了20帧就开始优化
std::vector<ImgOpt *> backend_buffer;
std::vector<poWeightL> weight_point;

std::unordered_map<int, std::vector<std::vector<int>>> color_map;
// 这个是多线程之间的映射，比如前端优化出来了一个frame，通过id寻找主线程中的该frame，就不需要遍历了
std::map<int, Frame *> frame_map;
std::vector<goMeas> opt_meas_buffer;
std::vector<BackendThread *> backend_opt_slider;
std::vector<FrontEndThread *> fe_opt_slider;
// 这个存在于一种情况：主线程循环了一次，frame_index自增了，但是这一帧的前端线程没发出去，那么就需要在后续补上
std::vector<int> opt_frame_index_buffer;
cv::Mat matrix_in;
cv::Mat matrix_out;
// cv::Mat camera_matrix;
// cv::Mat distortion_coef;
Eigen::Matrix3d matrixIn_eig;
Eigen::Matrix<double, 3, 4> matrixOut_eig;
Eigen::Isometry3d lidar_cam = Eigen::Isometry3d::Identity();
Eigen::Isometry3d before_opt_Tcw = Eigen::Isometry3d::Identity();

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
// 经过深度过滤的点云，采用同一区域深度最小的点放入优化过程
PointCloudXYZI::Ptr feats_undistort_filter(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort_world(new PointCloudXYZI());
// 局部点云，用来维护后端优化和以后的pub，用的是经过投影之后下采样的点云累加，而且所有点的坐标都处于世界坐标系
PointCloudXYZI::Ptr local_map_pc(new PointCloudXYZI());

PointCloudXYZRGB::Ptr cloudRGB(new PointCloudXYZRGB());
PointCloudXYZRGB::Ptr cloudRGBTemp(new PointCloudXYZRGB());
PointCloudXYZRGB::Ptr cloudRGB_down_body(new PointCloudXYZRGB());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Path path_camera;
nav_msgs::Odometry odomAftMapped;
nav_msgs::Odometry odomAftMapped_cam;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;
geometry_msgs::PoseStamped msg_body_pose_camera;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// ImgOpt last_Opt_cache;
double firstLidarFrame;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                            // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2));    // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2));    // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                 // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));       // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));       // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

inline void log3DPoint(Eigen::Vector3d point)
{
    std::cout << "x: " << point(0) << " y: " << point(1) << " z: " << point(2) << std::endl;
}

inline bool isOverlap(Eigen::Vector3d p1, Eigen::Vector3d p2)
{
    double r2 = filter_size_camera_opt * filter_size_camera_opt;
    // 因为livox的坐标系x轴是向外的，所以以y-z平面的距离构成半径判断，而深度x用于找到离我们最近的物体表面
    if (powl(p1.x() - p2.x(), 2) + powl(p1.y() - p2.y(), 2) <= r2)
    {
        return true;
    }

    return false;
}

// 返回世界坐标点转换到当前雷达局部坐标系的变换矩阵
inline Eigen::Isometry3d b2w_get()
{
    Eigen::Isometry3d body_2_world_1 = Eigen::Isometry3d::Identity();
    body_2_world_1.rotate(state_point.offset_R_L_I.toRotationMatrix());
    body_2_world_1.pretranslate(state_point.offset_T_L_I);
    Eigen::Isometry3d body_2_world_2 = Eigen::Isometry3d::Identity();
    body_2_world_2.rotate(state_point.rot);
    body_2_world_2.pretranslate(state_point.pos);
    Eigen::Isometry3d body_2_world = body_2_world_2 * body_2_world_1;
    return body_2_world;
}

// 不用矩阵乘法的原因是还得取元素，这么多次操作非常耗时，所以直接传值计算更省性能
inline Eigen::Vector2d proj3Dto2D(float x, float y, float z, float fx, float fy, float cx, float cy)
{
    float u = fx * x / z + cx;
    float v = fy * y / z + cy;
    return Eigen::Vector2d(u, v);
}

[[gnu::optimize(0)]] inline float getGrayScaleInImg(float u, float v, cv::Mat &img)
{
    float gray = -1.0;
    if (u < 0 || u > img.cols || v < 0 || v > img.rows)
    {
        return gray;
    }
    // 按照行列顺序来，所以u是横坐标，所以是col
    if (!img.empty())
    {
        gray = (float)img.at<uchar>(cvRound(v), cvRound(u));
    }
    return gray;
}

inline std::string timestampToDateString(std::time_t timestamp)
{
    std::chrono::system_clock::time_point timePoint = std::chrono::system_clock::from_time_t(timestamp);
    std::time_t time = std::chrono::system_clock::to_time_t(timePoint);
    std::tm localTime;
    gmtime_r(&time, &localTime);

    std::stringstream ss;
    ss << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");

    return ss.str();
}

inline cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv::Mat img;
    img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
    return img;
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointT const *const pi, PointT *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // state_point is IMU pose after optimized
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->r = pi->r;
    po->g = pi->g;
    po->b = pi->b;
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // state_point is IMU pose after optimized
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointT const *const pi, PointT *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->r = pi->r;
    po->g = pi->g;
    po->b = pi->b;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

void getOriginalCoordByTransMatrix(Quaterniond &rot, Eigen::Vector3d &ori, Eigen::Isometry3d transMatrix)
{

    Eigen::Vector3d orig_body(0, 0, 0);
    ori = transMatrix.inverse() * orig_body;
    rot = Quaterniond(transMatrix.rotation().inverse());
    rot.normalize();
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
double lidar_start_time_point = 0.0;
bool isInit = false;
bool isFIrstGetLidar = true;
// avia cbk func
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (isFIrstGetLidar)
    {
        firstLidarFrame = msg->header.stamp.toSec();
        isFIrstGetLidar = false;
    }
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    ROS_INFO("CURRENT received lidar point at time (msg):%lf", msg->header.stamp.toSec());
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    ROS_INFO("Get Lidar Cloud size --------------------------- %lu ", ptr->size());
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    ROS_INFO("time buffer size is :%lu", time_buffer.size());
    if (!isInit)
    {
        ROS_INFO("lidar_start_time_point is %lf ", last_timestamp_lidar);
        lidar_start_time_point = last_timestamp_lidar;
        isInit = true;
    }

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

int isFirst = 1;
void img_cbk(const sensor_msgs::ImageConstPtr &msg_in)
{

    if (msg_in->header.stamp.toSec() < last_timestamp_image)
    {
        ROS_ERROR("img loop back,clear buffer");
        img_buffer_pck.clear();
        img_time_buffer.clear();
    }

    mtx_buffer.lock();

    // initialImg.name = msg_in->header.frame_id;
    // initialImg.img_data = getImageFromMsg(msg_in);
    // initialImg.timestamp = msg_in->header.stamp.toSec();

    img_buffer_pck.push_back(getImageFromMsg(msg_in));
    img_time_buffer.push_back(msg_in->header.stamp.toSec());
    last_timestamp_image = msg_in->header.stamp.toSec();
    ROS_INFO("get img at  %lf", msg_in->header.stamp.toSec());
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    // IMU 的时间戳是从ros的时间中获取的
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
int sync_step = 0;
bool show_error_img_small_lidar_frame = true;

bool sync_packages_img(MeasureGroup &meas)
{
    ROS_ERROR("Image buffer size------%lu \n Image Time buffer size------%lu", img_buffer_pck.size(), img_time_buffer.size());
    if ((lidar_buffer.empty() && img_buffer_pck.empty()))
    { // has lidar topic or img topic?
        ROS_INFO("NO LIDAR POINT AND IMG");
        return false;
    }
    // ROS_ERROR("In sync");
    // if (meas.is_lidar_end) // If meas.is_lidar_end==true, means it just after scan end, clear all buffer in meas.
    // {
    //     // meas.measures.clear();
    //     meas.is_lidar_end = false;
    // }

    if (!lidar_pushed)
    { // If not in lidar scan, need to generate new meas
        ROS_INFO("starting sync lidar points!");
        if (lidar_buffer.empty())
        {
            ROS_ERROR("lidar_buffer empty");
            return false;
        }
        meas.lidar = lidar_buffer.front(); // push the firsrt lidar topic
        if (meas.lidar->points.size() <= 1)
        {
            ROS_INFO("this scan has too few points!");
            mtx_buffer.lock();
            if (img_buffer_pck.size() > 0) // temp method, ignore img topic when no lidar points, keep sync
            {
                ROS_INFO("while this scan's points are few,there has img in img_buffer,so discard them!");
                lidar_buffer.pop_front();
                img_buffer_pck.front().release();
                img_buffer_pck.pop_front();
                time_buffer.pop_front();
                img_time_buffer.pop_front();
            }
            mtx_buffer.unlock();
            sig_buffer.notify_all();
            ROS_ERROR("no lidar point in sync time");
            return false;
        }
        sort(meas.lidar->points.begin(), meas.lidar->points.end(), time_list); // sort by sample timestamp
        // ROS_INFO("LIDAR POINTS SIZE IS :%d", meas.lidar->points.size());
        meas.lidar_beg_time = time_buffer.front(); // generate lidar_beg_time
        ROS_INFO("Meas is at time :%lf", meas.lidar_beg_time);
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
        meas.lidar_end_time = lidar_end_time;
        ROS_INFO("the diff between start and end of a lidar scan is %lf", lidar_end_time - meas.lidar_beg_time);
        ROS_INFO("current lidar end time is :%lf", lidar_end_time);
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        lidar_pushed = true; // flag
    }

    if (last_timestamp_imu < lidar_end_time + 0.02)
    {
        ROS_ERROR("the LATEST IMU time is slower than current lidar frame!");
        return false;
    }

    if (!imu_pushed)
    {
        bool is_pushed_imu = false;
        if (imu_buffer.empty())
        {
            ROS_ERROR("IMU_buffer empty");
            return false;
        }
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        meas.imu.clear();
        mtx_buffer.lock();
        while ((!imu_buffer.empty() && (imu_time < lidar_end_time)))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if (imu_time > lidar_end_time)
            {
                ROS_ERROR("imu_buffer's elements are bigger than meas.time");
                break;
            }

            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
            is_pushed_imu = true;
        }
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        if (is_pushed_imu)
        {
            ROS_INFO("IMU SYNC SUCCESSED!");
            imu_pushed = true;
        }
        else
        {
            imu_pushed = false;
            lidar_pushed = false;
            ROS_INFO("IMU SYNC FAILED!");
            return false;
        }
    }

    if (!img_buffer_pck.empty())
    {
        // no img topic, means only has lidar topic
        // if the latest time stamp of a img is smaller than lidar_end_time - 0.5 of current lidar frame

        // 先判断最大粒度，即图片池的第一个是不是晚于lidar池的最新帧
        // 那么这个时候就得把前面的lidar都扔了，因为没用了，已经找不到对应的图片了
        // 随着程序运行的时间推移，img_buffer就算清空你的最新的图片肯定还是晚于雷达帧
        // 所以扔雷达帧是比较合理的，而不扔图片帧
        if (img_time_buffer.front() > lidar_end_time)
        {
            // 说明图片第一个采集的晚了，把前面的lidar frame仍了
            // 因为扔了，所以重新push
            imu_pushed = false;
            lidar_pushed = false;
            ROS_INFO("Reset the Measure of current frame");
            // mtx_buffer.lock();
            // // 通过while把早于img第一个的lidar帧都扔掉，这个可以在初始化的时候静止即可
            // while (!lidar_buffer.empty() && img_time_buffer.front() > lidar_end_time)
            // {
            //     ROS_INFO("Drop one lidar frame!");
            //     time_buffer.pop_front();
            //     lidar_buffer.pop_front();
            //     // 扔IMU是因为认为IMU与lidar的buffer差得不多，因为在rqt_bag里面显示IMU和Lidar的发布频率时二者对齐的不错
            //     imu_buffer.pop_front();
            // }
            // mtx_buffer.unlock();
            // sig_buffer.notify_all();
            // 思路是一直扔，扔到图片帧小于雷达帧，这时候就可以同步了，但是前面的lidar和imu要重新同步，也就是说把标志位都复原
            // 且由于考虑到一旦雷达扔完了，之后的照片和雷达其实发布频率比较一致，所以对齐问题应该不大
            return false;
        }

        double img_start_time = img_time_buffer.front();
        meas.img_offset_time = -1.0;
        ROS_INFO("img start time at %lf,lidar end time at %lf", img_start_time, lidar_end_time);
        mtx_buffer.lock();
        while ((!img_buffer_pck.empty()) && img_start_time < lidar_end_time)
        {
            ROS_INFO("CHOOSE IMG START!");
            img_start_time = img_time_buffer.front();
            if (img_start_time < meas.lidar_beg_time)
            {
                ROS_INFO("this img start time = %lf || this lidar frame end time = %lf", img_start_time, lidar_end_time);
                img_buffer_pck.front().release();
                img_buffer_pck.pop_front();
                img_time_buffer.pop_front();
                // 在这里也要更新一下，不然下次依然是上一个没pop的time，继续进，如果遇到time buffer里面和上一个img time相差很大的，就会出错
                img_start_time = img_time_buffer.front();
                continue;
            }
            meas.img = img_buffer_pck.front();
            meas.img_offset_time = img_start_time - meas.lidar_beg_time; // record img offset time, it shoule be the Kalman update timestamp.
            img_buffer_pck.front().release();
            img_buffer_pck.pop_front();
            img_time_buffer.pop_front();
            ROS_INFO("Get Img at %lf with Sync success!", meas.img_offset_time + meas.lidar_beg_time);
            break;
        }
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        if (meas.img_offset_time == -1.0)
        {
            return false;
        }
        // lidar_buffer.pop_front();
        // time_buffer.pop_front();
        lidar_pushed = false;     // sync one whole lidar scan.
        imu_pushed = false;       // means imu,lidar and img are all sync completely.
        meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
        ROS_INFO("LIDAR SYNC!!!");
        ROS_WARN("in sync meas pck,lidar frame is :%lf || IMU frame is :%lf || img start time is :%lf", meas.lidar_beg_time, meas.imu.front()->header.stamp.toSec(),
                 meas.img_offset_time + meas.lidar_beg_time);
        // ROS_ERROR("out sync");
        return true;
    }
    return false;
}
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }
    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {

        meas.lidar = lidar_buffer.front();
        ROS_INFO("this scan has %lu points", meas.lidar->points.size());
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

// PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZRGB::Ptr pcl_wait_save_rgb(new PointCloudXYZRGB());
float UV[2] = {0, 0};
// use extrinsic and intrinsic to get the corresponding U and V
void getUV(const Quaterniond &deltaQ, const V3D deltaPos, const cv::Mat &matrix_in, const cv::Mat &matrix_out, float x, float y, float z, float *UV)
{

    V3D pos_in_lidar(x, y, z);
    Matrix3d rot_l_cam = deltaQ.toRotationMatrix();
    V3D pos_in_image = rot_l_cam * pos_in_lidar + deltaPos;
    //   V3D pos_in_image = deltaQ * pos_in_lidar;
    double matrix3[4][1] = {pos_in_image.x(), pos_in_image.y(), pos_in_image.z(), 1};
    // double matrix3[4][1] = {pos_in_lidar.x(), pos_in_lidar.y(), pos_in_lidar.z(), 1};
    cv::Mat coordinate(4, 1, CV_64F, matrix3);

    // calculate the result of u and v
    // cv::Mat result = matrix_in*matrix_out*coordinate;
    cv::Mat result = matrix_in * matrix_out * (coordinate);
    float u = result.at<double>(0, 0);
    float v = result.at<double>(1, 0);
    float depth = result.at<double>(2, 0);

    UV[0] = u / depth;
    UV[1] = v / depth;

    if (first_first_point)
    {
        first_first_point = false;
        std::cout << "Before opt the first" << std::endl;
        std::cout << "point body" << std::endl;
        log3DPoint(pos_in_lidar);
        std::cout << "point cam" << std::endl;
        log3DPoint(pos_in_image);
        std::cout << "pixel U:" << UV[0] << " V:" << UV[1] << std::endl;
    }
}
void getUV(const Eigen::Isometry3d &tcw, Eigen::Vector3d p_world, float *UV)
{

    // V3D pos_in_world(x, y, z);
    V3D pos_in_image = tcw * p_world;
    //   V3D pos_in_image = deltaQ * pos_in_lidar;
    double matrix3[3][1] = {pos_in_image.x(), pos_in_image.y(), pos_in_image.z()};
    // double matrix3[4][1] = {pos_in_lidar.x(), pos_in_lidar.y(), pos_in_lidar.z(), 1};
    // 这个已经是相机坐标系坐标了，所以只需要内参就能转换了
    cv::Mat coordinate(3, 1, CV_64F, matrix3);

    // calculate the result of u and v
    // cv::Mat result = matrix_in*matrix_out*coordinate;
    cv::Mat result = matrix_in * (coordinate);
    float u = result.at<double>(0, 0);
    float v = result.at<double>(1, 0);
    float depth = result.at<double>(2, 0);

    UV[0] = u / depth;
    UV[1] = v / depth;

    if (0)
    {
        first_first_point = false;
        std::cout << "Before opt the first" << std::endl;
        std::cout << "point body" << std::endl;
        log3DPoint(p_world);
        std::cout << "point cam" << std::endl;
        log3DPoint(pos_in_image);
        std::cout << "pixel U:" << UV[0] << " V:" << UV[1] << std::endl;
    }
}
void getColor(const Quaterniond &deltaQ, const V3D deltaPos, const cv::Mat &matrix_in, const cv::Mat &matrix_out, float x, float y, float z, int row, int col, const std::vector<std::vector<int>> &color_vector, int *RGB)
{
    UV[0] = 0.0;
    UV[1] = 0.0;
    getUV(deltaQ, deltaPos, matrix_in, matrix_out, x, y, z, UV); // get U and V from the x,y,z

    int u = int(UV[0]);
    int v = int(UV[1]);

    int32_t index = v * col + u;

    if (index < row * col && index >= 0)
    {
        RGB[0] = color_vector[index][0];
        RGB[1] = color_vector[index][1];
        RGB[2] = color_vector[index][2];
    }
    else
    {
        RGB[0] = -1;
        RGB[1] = -1;
        RGB[2] = -1;
    }
}
void getColor(const Eigen::Isometry3d tcw, Eigen::Vector3d p_world, int row, int col, const std::vector<std::vector<int>> &color_vector, int (&RGB)[3])
{
    UV[0] = 0.0;
    UV[1] = 0.0;
    getUV(tcw, p_world, UV); // get U and V from the x,y,z

    int u = int(UV[0]);
    int v = int(UV[1]);

    int32_t index = v * col + u;

    if (index < row * col && index >= 0)
    {
        RGB[0] = color_vector[index][0];
        RGB[1] = color_vector[index][1];
        RGB[2] = color_vector[index][2];
    }
    else
    {
        RGB[0] = -1;
        RGB[1] = -1;
        RGB[2] = -1;
    }
}
void publish_frame_world(const ros::Publisher &pubLaserCloudFull)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;
        times_pub++;
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

V3D convertRos2EigenVector(const nav_msgs::Odometry o)
{
    V3D temp;
    temp << o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z;
    return temp;
}

Quaterniond convertRos2Eigen(const nav_msgs::Odometry o1)
{
    Quaterniond temp(o1.pose.pose.orientation.w, o1.pose.pose.orientation.x, o1.pose.pose.orientation.y, o1.pose.pose.orientation.z);
    return temp;
}

V3D pos_slerp(double t, V3D p1, V3D p2)
{
    V3D temp;
    temp = (1 - t) * p1 + t * p2;
    return temp;
}

Quaterniond slerp(double t, Quaterniond &q1, Quaterniond &q2)
{
    // ---- 开始你的代码 ----- //
    q1.normalize();
    q2.normalize();
    // 四元数变为Mat: 方法 1
    //    double q1_array[4] = { q1.w(), q1.x(), q1.y(), q1.z()};
    //    double q2_array[4] = { q2.w(), q2.x(), q2.y(), q2.z()};
    //    Mat q1_Mat(4,1,CV_64F, q1_array);
    //    Mat q2_Mat(4, 1, CV_64F, q2_array);

    // 四元数变为Mat: 方法 2
    cv::Mat q1_Mat = (cv::Mat_<double>(4, 1) << q1.w(), q1.x(), q1.y(), q1.z());
    cv::Mat q2_Mat = (cv::Mat_<double>(4, 1) << q2.w(), q2.x(), q2.y(), q2.z());
    double dotProd = q1_Mat.dot(q2_Mat);                                          // q1与q2的点积
    double norm2 = cv::norm(q1_Mat, cv::NORM_L2) * cv::norm(q2_Mat, cv::NORM_L2); // L2为2范数 即q1的模乘以q2的模
    double cosTheta = dotProd / norm2;                                            // cosθ = q1.*q2/(||q1||+||q2||)

    cv::Mat result;
    Quaterniond result_quat;
    if (cosTheta > 0.9995f) // 如果θ太小 就使用一元线性插值
    {
        result = (1.0f - t) * q1_Mat + t * q2_Mat;
    }
    else // 否则就使用球面线性插值的简化方法：v'=v1*cosθ' + v⊥*sinθ'
    {
        double theta = acosf(cosTheta);
        double thetaT = theta * t; // t为0-1的小数 thetaT即为 θ‘
        // q1 q2都为向量，现在要求q1的垂直向量qperp
        // 把q2进行向量分解 q2=qperp*sinθ + q1*cosθ
        // 解出qperp
        cv::Mat qperp = (q2_Mat - cosTheta * q1_Mat) / sinf(theta); // qperp即为V⊥，即q1的垂直向量
        result = q1_Mat * cosf(thetaT) + qperp * sinf(thetaT);
    }

    result = result / cv::norm(result, cv::NORM_L2);
    // Mat 转化为四元数
    result_quat.w() = result.at<double>(0, 0);
    result_quat.x() = result.at<double>(1, 0);
    result_quat.y() = result.at<double>(2, 0);
    result_quat.z() = result.at<double>(3, 0);

    return result_quat;
    // ---- 结束你的代码 ----- //
}

// 使用球面插值去模拟出每20帧之间的图片偏移位姿然后补偿给映射的偏移计算
// 计算出位姿（方向和位置）之后，如何补偿？因为是世界坐标系下的坐标，所以直接传回
nav_msgs::Odometry odomAftInterpolation(double t, double t1, float t2, nav_msgs::Odometry o1, nav_msgs::Odometry o2)
{
    ROS_WARN("Slerp in %d-th pub", times_pub);

    if ((o1.pose.pose.orientation == o2.pose.pose.orientation && o1.pose.pose.position == o2.pose.pose.position) || t1 == t2)
    {
        ROS_WARN("Equal!");
        return o2;
    }
    // 当传进来的时间小于0，说明前一帧时间不存在，即是初始化时期，只有一帧，就把第一帧的位姿传给照片，认为二者是对齐的
    if (t1 < 0 || t2 < 0)
    {
        ROS_WARN("TIME ERROR %lf, %lf", t1, t2);
        return o2;
    }

    // double t_temp = (t - t1) / (t2 - t1); // 计算出t在两个雷达时间戳中的比例
    double t_temp = (t - t1) / (t2 - t1);
    Quaterniond q1 = convertRos2Eigen(o1);
    Quaterniond q2 = convertRos2Eigen(o2);
    q1.normalize();
    q2.normalize();
    Eigen::Vector3d p1;
    p1 << o1.pose.pose.position.x, o1.pose.pose.position.y, o1.pose.pose.position.z;
    Eigen::Vector3d p2;
    p2 << o2.pose.pose.position.x, o2.pose.pose.position.y, o2.pose.pose.position.z;

    Quaterniond oriAftLerp = slerp(t_temp, q1, q2);
    oriAftLerp.normalize();
    Eigen::Vector3d posAftLerp = pos_slerp(t_temp, p1, p2);
    // 此时得到的位置和朝向是20帧之间的某一帧的相机的位姿，而这个位姿就用于将照片变换到世界坐标系来投影
    nav_msgs::Odometry odomAftLerp;
    odomAftLerp.pose.pose.position.x = posAftLerp.x();
    odomAftLerp.pose.pose.position.y = posAftLerp.y();
    odomAftLerp.pose.pose.position.z = posAftLerp.z();

    odomAftLerp.pose.pose.orientation.w = oriAftLerp.w();
    odomAftLerp.pose.pose.orientation.x = oriAftLerp.x();
    odomAftLerp.pose.pose.orientation.y = oriAftLerp.y();
    odomAftLerp.pose.pose.orientation.z = oriAftLerp.z();

    return odomAftLerp;
}

void RGBPoint_body2slerpBody(PointT const *const pi, PointT *const po, const V3D &deltaPos, const Quaterniond &deltaQ)
{
    V3D p_body(pi->x, pi->y, pi->z);
    Matrix3d rot = deltaQ.toRotationMatrix();
    V3D p_body_slerpBody(rot * p_body + deltaPos);
    // Matrix3d rot_l_cam = deltaQ.toRotationMatrix();
    // V3D pos_in_image = rot_l_cam * pos_in_lidar + deltaPos;
    //  V3D pos_in_image = deltaQ * pos_in_lidar;
    // double matrix3[4][1] = {pos_in_image.x(), pos_in_image.y(), pos_in_image.z(), 1};
    po->x = p_body_slerpBody(0);
    po->y = p_body_slerpBody(1);
    po->z = p_body_slerpBody(2);
}
// define it to optimize the pose at a certain state (using ceres solver)
// return an optimized state(pose and orientation)
// using optical flow to compute out the pose instead?
// 1. we got 2 frame of image,so the transformation from A to B could be computed by optical flow
// 2. once we got the pose we interploate,using this to supervise the simulated pose
// 3. Is the result of step2 could bbe recognized as GT?
//  nav_msgs::Odometry optimize_by_photometric(cv::Mat pre_frame,cv::Mat cur_frame,){

// }

// 记录发布的帧数

nav_msgs::Odometry odomCache_msg;
nav_msgs::Odometry currentCamerPose;
// l1与l3插值l2，然后进行对比
double preLidarFrame = 0;
bool isChanged = false;
int min_img_index = 0;
bool isInitial_frame = true;
int count_down = 1;
cv::RNG rng;
void publish_frame_world_rgb(const ros::Publisher &pubLaserCloudFull, vector<Frame *> &backend_opt_buffer, PointCloudXYZI::Ptr map)
{
    // PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    // PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);

    // local_map_pc就是局部的地图点，里面都是世界坐标
    PointCloudXYZI::Ptr laserCloudFullRes(new PointCloudXYZI());
    pcl::copyPointCloud(*map, *laserCloudFullRes);
    // PointCloudT::Ptr laserCloudFullRes(cloudRGB);
    int pose_ssize = backend_opt_buffer.size();
    invalid_count = 0;

    // std::unordered_map<int, std::vector<std::vector<int>>>().swap(color_map);
    // //color_map.clear();
    // for (int i = 0; i < pose_ssize; i++)
    // {
    //     vector<vector<int>> color_table;
    //     cv::Mat temp = backend_opt_buffer[i].img_ref;
    //     int row = temp.rows;
    //     int col = temp.cols;
    //     color_table.resize(row * col, vector<int>(3));
    //     for (int v = 0; v < row; ++v)
    //     {
    //         for (int u = 0; u < col; ++u)
    //         {
    //             // Vec3b是有根据的，需要根据图片的通道或者矩阵类型来
    //             int r = temp.at<cv::Vec3b>(v, u)[2];
    //             int g = temp.at<cv::Vec3b>(v, u)[1];
    //             int b = temp.at<cv::Vec3b>(v, u)[0];
    //             color_table[v * col + u][0] = r;
    //             color_table[v * col + u][1] = g;
    //             color_table[v * col + u][2] = b;
    //         }
    //     }
    //     color_map.insert(std::pair<int, std::vector<std::vector<int>>>(i, color_table));
    // }
    // ROS_WARN("COLOR MAP COUNT %d,COLOR BUFFER ONE FRAME %d",color_map.size(),color_map[0].size());
    int size = laserCloudFullRes->points.size();
    PointCloudT::Ptr cloudRGBTemps(new PointCloudT());
    cloudRGBTemps->clear();
    cloudRGBTemps->is_dense = true;
    cloudRGBTemps->height = 1;

    for (int i = 0; i < size; i++)
    {

        vector<Eigen::VectorXd> all_color;

        Eigen::Vector3d p_world(laserCloudFullRes->points[i].x, laserCloudFullRes->points[i].y, laserCloudFullRes->points[i].z);
        for (int j = 0; j < pose_ssize; j++)
        {
            int RGB_temp[3] = {-1};
            // Eigen::Vector2d uv = proj3Dto2D(temp(0), temp(1), temp(2), matrixIn_eig(0, 0), matrixIn_eig(1, 1), matrixIn_eig(0, 2), matrixIn_eig(1, 2));
            int row = backend_opt_buffer[j]->image().rows;
            int col = backend_opt_buffer[j]->image().cols;
            Eigen::Vector3d point_cam = backend_opt_buffer[j]->pose() * p_world;
            Eigen::Vector2d pixels = proj3Dto2D(point_cam.x(), point_cam.y(), point_cam.z(), matrixIn_eig(0, 0), matrixIn_eig(1, 1), matrixIn_eig(0, 2), matrixIn_eig(1, 2));
            // getColor(backend_opt_buffer[j].lidar_img,p_world, row, col, color_map.at(j),RGB_temp);
            if (pixels(0) >= 0 && pixels(0) <= col && pixels(1) >= 0 && pixels(1) <= row)
            {
                RGB_temp[0] = backend_opt_buffer[j]->image().at<cv::Vec3b>(pixels(1), pixels(0))[2];
                RGB_temp[1] = backend_opt_buffer[j]->image().at<cv::Vec3b>(pixels(1), pixels(0))[1];
                RGB_temp[2] = backend_opt_buffer[j]->image().at<cv::Vec3b>(pixels(1), pixels(0))[0];
            }

            if (RGB_temp[0] == -1 && RGB_temp[1] == -1 && RGB_temp[2] == -1)
            {
                valid_orig_count++;
                continue;
            }
            Eigen::VectorXd temp_rgb_v3d(3);
            temp_rgb_v3d << RGB_temp[0], RGB_temp[1], RGB_temp[2];
            all_color.push_back(temp_rgb_v3d);
        }
        Eigen::MatrixXd color_matrix;
        int color_mat_rows = all_color.size();
        if (color_mat_rows == 0)
        {
            continue;
        }
        color_matrix.resize(color_mat_rows, 3);
        for (int k = 0; k < color_mat_rows; k++)
        {
            color_matrix.row(k) = all_color[k];
        }
        Eigen::VectorXd av_color = color_matrix.colwise().mean();
        PointT rgb_point;
        rgb_point.x = p_world(0);
        rgb_point.y = p_world(1);
        rgb_point.z = p_world(2);
        rgb_point.r = av_color(0);
        rgb_point.g = av_color(1);
        rgb_point.b = av_color(2);
        cloudRGBTemps->push_back(rgb_point);
    }
    for (int i = 0; i < pose_ssize; i++)
    {
        backend_opt_buffer[i]->setPublishedStatus(true);
    }
    // for (int i = 0; i < size; i++)
    // {
    //     RGBpointBodyToWorld(&cloudRGB->points[i], &cloudRGBWorld->points[i]);
    //     // std::cout<<"world pos =("<<cloudRGBWorld->points[i].x<<","<<cloudRGBWorld->points[i].y<<","<<cloudRGBWorld->points[i].z<<","<<cloudRGBWorld->points[i].r<<","<<cloudRGBWorld->points[i].g<<")";
    //     // ROS_INFO_STREAM("world pos =("<<cloudRGBWorld->points[i].x<<","<<cloudRGBWorld->points[i].y<<","<<cloudRGBWorld->points[i].z<<","<<cloudRGBWorld->points[i].r<<","<<cloudRGBWorld->points[i].g<<")");
    // }
    // optpub
    // ROS_INFO("This pub has %d white color but we have %d points in one scan!", a_count, size);
    // sensor_msgs::PointCloud2 laserCloudmsg;
    // pcl::toROSMsg(*cloudRGBTemps, laserCloudmsg);
    // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    // laserCloudmsg.header.frame_id = "camera_init";
    // pubLaserCloudFull.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;

    //}
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    // pcd_save_en
    if (pcd_save_en)
    {
        *pcl_wait_save_rgb += *cloudRGBTemps;
        ROS_WARN("Add point to world map,%lu", cloudRGBTemps->size());
        ROS_WARN("Now the pcl to be saved has %lu points!", pcl_wait_save_rgb->size());
        static int scan_wait_num = 0;
        scan_wait_num++;
        // && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval
        if (pcl_wait_save_rgb->size() > rgb_point_save_num)
        {
            ROS_INFO("STORE POINTS IN WHILE LOOP,pcl_rgb's size is :%lu", pcl_wait_save_rgb->size());
            pcd_index_rgb++;
            string file_name = string("scans_rgb_" + dateString + string("_") + to_string(pcd_index_rgb) + string(".pcd"));
            string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            // string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_rgb") + to_string(pcd_index_rgb) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_rgb);
            pcl_wait_save_rgb->clear();
            scan_wait_num = 0;
        }
    }
    map->clear();
    map.reset();
    cloudRGBTemps->clear();
    cloudRGBTemps.reset();
}

bool initial_frame = true;
Quaterniond deltaQ;
V3D deltaPos;
int store_log_img = 5;
vector<vector<int>> color_vector;
void publish_frame_body_rgb(const ros::Publisher &pubLaserCloudFull_body)
{
    ROS_INFO("==================PUBLIC BODY RGB POINTS START==================");
    valid_orig_count = 0;
    first_first_point = true;
    ROS_INFO("the %d -th pub", times_pub);

    ROS_INFO("current time buffer size: %lu", time_buffer.size());

    for (int i = 0; i < time_buffer.size(); i++)
    {
        ROS_INFO("the %d-th time is : %lf", i, time_buffer[i]);
    }

    // PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
    int size = laserCloudFullRes->points.size();
    ROS_INFO("------------------- %d", size);
    // 此时已经得到了两帧之间的插值，那么肯定是转回前一帧

    // Quaterniond lerp_q = convertRos2Eigen(currentCamerPose);
    cv::Mat src_img = Measures.img;
    // cv::Mat view, rview, map1, map2;
    // cv::Size imageSize = src_img.size();
    // cv::initUndistortRectifyMap(camera_matrix, distortion_coef, cv::Mat(), cv::getOptimalNewCameraMatrix(camera_matrix, distortion_coef, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, map1, map2);
    // cv::remap(src_img, src_img, map1, map2, cv::INTER_LINEAR); // correct the distortion
    if (src_img.empty())
    {
        ROS_INFO("empty error image!\n");
        return;
    }

    row = src_img.rows;
    col = src_img.cols;
    color_vector.resize(row * col, vector<int>(3));
    ROS_INFO("color vector size is : %lu", color_vector.size());
    ROS_INFO("Start to read the photo ");
    for (int v = 0; v < row; ++v)
    {
        for (int u = 0; u < col; ++u)
        {
            // Vec3b是有根据的，需要根据图片的通道或者矩阵类型来
            int r = src_img.at<cv::Vec3b>(v, u)[2];
            int g = src_img.at<cv::Vec3b>(v, u)[1];
            int b = src_img.at<cv::Vec3b>(v, u)[0];
            color_vector[v * col + u][0] = r;
            color_vector[v * col + u][1] = g;
            color_vector[v * col + u][2] = b;
        }
    }
    ROS_INFO("Finish saving the data ");

    //-----------------验证l2 frame的代码插在这个下面-------------------------//

    // ROS_INFO("Delta w:%f,x:%f,y:%f,z:%f-------------DeltaPos x:%f,y:%f,z:%f", deltaQ.w(), deltaQ.x(), deltaQ.y(), deltaQ.z(), deltaPos.x(), deltaPos.y(), deltaPos.z());

    int RGB[3] = {0, 0, 0};
    cloudRGB->clear();
    cloudRGB->is_dense = true;
    cloudRGB->height = 1;
    cloudRGB->width = size; // get the point number of lidar data
    cloudRGB->points.resize(size);

    std::vector<cv::Point2f>().swap(cache_lidar_2_img);
    cv::Mat circle_img = src_img.clone();
    // PointCloudT::Ptr cloudRGBTemp(new PointCloudT);
    cloudRGBTemp->clear();
    cloudRGBTemp->is_dense = true;
    cloudRGBTemp->height = 1;
    for (int i = 0; i < size; i++)
    {

        cloudRGB->points[i].x = (&laserCloudFullRes->points[i])->x;
        cloudRGB->points[i].y = (&laserCloudFullRes->points[i])->y;
        cloudRGB->points[i].z = (&laserCloudFullRes->points[i])->z;
        getColor(deltaQ, deltaPos, matrix_in, matrix_out, (&laserCloudFullRes->points[i])->x, (&laserCloudFullRes->points[i])->y, (&laserCloudFullRes->points[i])->z, row, col, color_vector, RGB);
        if (RGB[0] == -1 && RGB[1] == -1 && RGB[2] == -1)
        {
            valid_orig_count++;
            continue;
        }
        cloudRGB->points[i].r = RGB[0];
        cloudRGB->points[i].g = RGB[1];
        cloudRGB->points[i].b = RGB[2];
        cloudRGBTemp->push_back(cloudRGB->points[i]);
        cache_lidar_2_img.push_back(cv::Point2f(UV[0], UV[1]));
    }
    before_opt_valid_ratio = (float)valid_orig_count / (float)size;
    if (cache_lidar_2_img.size() == 0)
    {
        ROS_INFO("In %d-th pub,no point porjected onto corresponding IMG", times_pub);
    }
    if (frame_map_to_save >= 0)
    {
        for (int i = 0; i < cache_lidar_2_img.size(); i++)
        {
            cv::Mat_<int> fillRNG(1, 3);
            rng.fill(fillRNG, cv::RNG::UNIFORM, 0, 256);
            cv::circle(circle_img, cache_lidar_2_img[i], 3, cv::Scalar_<int>(fillRNG[0][0], fillRNG[0][1], fillRNG[0][2]), 1);
        }
        cv::imwrite("/opt/log_img/bodypub/" + to_string(frame_map_to_save) + ".jpg", circle_img);
        // frame_map_to_save--;
    }

    PointCloudT::Ptr cloudRGBWorld(new PointCloudT);
    cloudRGBWorld->is_dense = true;
    cloudRGBWorld->height = 1;
    cloudRGBWorld->width = cloudRGBTemp->size(); // get the point number of lidar data
    // cloudRGBWorld->points.resize(cloudRGBTemp->size());
    //  PointCloudXYZRGB::Ptr cloudRGBWorld(new PointCloudXYZRGB(size,1));
    for (int i = 0; i < cloudRGBWorld->size(); i++)
    {
        RGBpointBodyLidarToIMU(&cloudRGBTemp->points[i], &cloudRGBWorld->points[i]);
        // std::cout<<"world pos =("<<cloudRGBWorld->points[i].x<<","<<cloudRGBWorld->points[i].y<<","<<cloudRGBWorld->points[i].z<<","<<cloudRGBWorld->points[i].r<<","<<cloudRGBWorld->points[i].g<<")";
        // ROS_INFO_STREAM("world pos =("<<cloudRGBWorld->points[i].x<<","<<cloudRGBWorld->points[i].y<<","<<cloudRGBWorld->points[i].z<<","<<cloudRGBWorld->points[i].r<<","<<cloudRGBWorld->points[i].g<<")");
    }

    ROS_INFO("This pub has %d white color but we have %d points in one scan!", valid_orig_count, size);
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*cloudRGBWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    simulate_time_lidar += 0.1;
    publish_count -= PUBFRAME_PERIOD;
    times_pub++;

    //}
    ROS_INFO("==================PUBLIC BODY RGB POINTS END==================");
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    // 这一段可以看出state_point代表的就是世界坐标系下的信息
    // geouat也一样，代表该点相对于世界坐标系的旋转，用四元数表示的
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
}
template <typename T>
void set_posestamp_camera(T &out)
{
    // 这一段可以看出state_point代表的就是世界坐标系下的信息
    // geouat也一样，代表该点相对于世界坐标系的旋转，用四元数表示的
    out.pose.position.x = currentCamerPose.pose.pose.position.x;
    out.pose.position.y = currentCamerPose.pose.pose.position.y;
    out.pose.position.z = currentCamerPose.pose.pose.position.z;
    out.pose.orientation.x = currentCamerPose.pose.pose.orientation.x;
    out.pose.orientation.y = currentCamerPose.pose.pose.orientation.y;
    out.pose.orientation.z = currentCamerPose.pose.pose.orientation.z;
    out.pose.orientation.w = currentCamerPose.pose.pose.orientation.w;
}

template <typename T>
void set_posestamp_camera(T &out, Eigen::Quaterniond rot, Eigen::Vector3d pose)
{
    // 这一段可以看出state_point代表的就是世界坐标系下的信息
    // geouat也一样，代表该点相对于世界坐标系的旋转，用四元数表示的
    out.pose.position.x = pose.x();
    out.pose.position.y = pose.y();
    out.pose.position.z = pose.z();
    out.pose.orientation.x = rot.x();
    out.pose.orientation.y = rot.y();
    out.pose.orientation.z = rot.z();
    out.pose.orientation.w = rot.w();
}

void publish_odometry_camera_backend(const ros::Publisher &pubOdomAftMapped, Eigen::Isometry3d &opt_T)
{
    odomAftMapped_cam.header.frame_id = "camera_init";
    odomAftMapped_cam.child_frame_id = "bodys";
    odomAftMapped_cam.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);

    Eigen::Quaterniond rot_temp;
    Eigen::Vector3d trans_temp;

    getOriginalCoordByTransMatrix(rot_temp, trans_temp, opt_T);

    set_posestamp_camera(odomAftMapped_cam.pose, rot_temp, trans_temp);
    pubOdomAftMapped.publish(odomAftMapped_cam);

    // auto P = kf.get_P();
    // for (int i = 0; i < 6; i++)
    // {
    //     int k = i < 3 ? i + 3 : i - 3;
    //     odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    //     odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    //     odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    //     odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    //     odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    //     odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    // }
    static tf::TransformBroadcaster br;
    tf::Transform transform_s;
    tf::Quaternion q_s;
    transform_s.setOrigin(tf::Vector3(odomAftMapped_cam.pose.pose.position.x,
                                      odomAftMapped_cam.pose.pose.position.y,
                                      odomAftMapped_cam.pose.pose.position.z));
    q_s.setW(odomAftMapped_cam.pose.pose.orientation.w);
    q_s.setX(odomAftMapped_cam.pose.pose.orientation.x);
    q_s.setY(odomAftMapped_cam.pose.pose.orientation.y);
    q_s.setZ(odomAftMapped_cam.pose.pose.orientation.z);
    transform_s.setRotation(q_s);
    br.sendTransform(tf::StampedTransform(transform_s, odomAftMapped_cam.header.stamp, "camera_init", "bodys"));
}

void publish_odometry_camera(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped_cam.header.frame_id = "camera_init";
    odomAftMapped_cam.child_frame_id = "bodys";
    odomAftMapped_cam.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);

    Eigen::Quaterniond rot_temp;
    Eigen::Vector3d trans_temp;

    getOriginalCoordByTransMatrix(rot_temp, trans_temp, multi_opt_buffer.back()->lidar_img);

    set_posestamp_camera(odomAftMapped_cam.pose, rot_temp, trans_temp);
    pubOdomAftMapped.publish(odomAftMapped_cam);

    // auto P = kf.get_P();
    // for (int i = 0; i < 6; i++)
    // {
    //     int k = i < 3 ? i + 3 : i - 3;
    //     odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
    //     odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
    //     odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
    //     odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
    //     odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
    //     odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    // }
    static tf::TransformBroadcaster br;
    tf::Transform transform_s;
    tf::Quaternion q_s;
    transform_s.setOrigin(tf::Vector3(odomAftMapped_cam.pose.pose.position.x,
                                      odomAftMapped_cam.pose.pose.position.y,
                                      odomAftMapped_cam.pose.pose.position.z));
    q_s.setW(odomAftMapped_cam.pose.pose.orientation.w);
    q_s.setX(odomAftMapped_cam.pose.pose.orientation.x);
    q_s.setY(odomAftMapped_cam.pose.pose.orientation.y);
    q_s.setZ(odomAftMapped_cam.pose.pose.orientation.z);
    transform_s.setRotation(q_s);
    br.sendTransform(tf::StampedTransform(transform_s, odomAftMapped_cam.header.stamp, "camera_init", "bodys"));
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    ROS_INFO("==================PUBLIC ODOMETRY START==================");
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform_s;
    tf::Quaternion q_s;

    transform_s.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                      odomAftMapped.pose.pose.position.y,
                                      odomAftMapped.pose.pose.position.z));
    q_s.setW(odomAftMapped.pose.pose.orientation.w);
    q_s.setX(odomAftMapped.pose.pose.orientation.x);
    q_s.setY(odomAftMapped.pose.pose.orientation.y);
    q_s.setZ(odomAftMapped.pose.pose.orientation.z);
    transform_s.setRotation(q_s);
    br.sendTransform(tf::StampedTransform(transform_s, odomAftMapped.header.stamp, "camera_init", "body"));
    ROS_INFO("==================PUBLIC ODOMETRY END==================");
}

void publish_path(const ros::Publisher &pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_path_interpolate_backend(const ros::Publisher &pubPath, Eigen::Isometry3d &opt_T)
{
    Eigen::Quaterniond rot_temp;
    Eigen::Vector3d trans_temp;

    getOriginalCoordByTransMatrix(rot_temp, trans_temp, opt_T);

    set_posestamp_camera(msg_body_pose_camera, rot_temp, trans_temp);
    msg_body_pose_camera.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose_camera.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int kkk = 0;
    kkk++;
    if (kkk % 10 == 0)
    {
        path_camera.poses.push_back(msg_body_pose_camera);
        pubPath.publish(path_camera);
    }
}

void publish_path_interpolate(const ros::Publisher &pubPath)
{
    Eigen::Quaterniond rot_temp;
    Eigen::Vector3d trans_temp;

    getOriginalCoordByTransMatrix(rot_temp, trans_temp, multi_opt_buffer.back()->lidar_img);

    set_posestamp_camera(msg_body_pose_camera, rot_temp, trans_temp);
    msg_body_pose_camera.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose_camera.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int kkk = 0;
    kkk++;
    if (kkk % 1 == 0)
    {
        path_camera.poses.push_back(msg_body_pose_camera);
        pubPath.publish(path_camera);
    }
}

// 第二个参数代表ieskf里面的dyn_share
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        // 如果收敛就进该代码块
        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // 存储第几个点是否满足要求
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        // 如果不满足条件，就跳出去继续判断下一个特征点
        if (!point_selected_surf[i])
            continue;

        VF(4)
        pabcd;
        point_selected_surf[i] = false;
        // 计算该平面的法向量
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                // norm_vec的点云数据结构里面存的都是某特征点对应的平面的法向量
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                // 残差存入intensity
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        // 为true代表该特征点有效，因为第i个特征点和point_selected_surf[i]是一一对应的
        if (point_selected_surf[i])
        { /*考虑一下没有对齐的情况
              即i-th的点并不是effect point，所以effct_feat_num不增加
              那么下一个i就会对应上一个effct_feat_num
              比如i=0，对不上，i=1，对上了，那么effct_feat_num=0
              不过，总体下来，laserCloudOri有值的部分还是effct_feat_num的大小
              也就是说laserCloudOri里面存的都是有效点的自身坐标系下的坐标
          */

            // 存的是下采样中的有效点，其坐标系还是自身坐标系
            // 注意：laserCloudOri的size和point_selected_surf的size都是10000，所以不会出现什么溢出或者匹配不上的情况
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            // 存的是有效点的对应的地图平面的法向量
            // corr_normvect里存的点也是与有效特征点一一对应的
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/

    // ekfom_data是我最后要输出给 “update_iterated_dyn_share_modified” 函数使用的数据，所以在本函数里计算
    // h_x 为观测相对于（姿态、位置、imu和雷达间的变换）
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
    // h就是观测方程吧
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        // 把有效点变成反对称矩阵
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        // offset_R_L_I，offset_T_L_I为IMU的旋转姿态和位移,此时转到了IMU坐标系下
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            // h_x的格式应该是（m，12）的矩阵，m代表特征点数量，也即观测纬度
            // h_x[0]=[n_x,n_y,n_z,]
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        // 在这一步初始化h，存的是到平面的距离（其实不就是观测信息么？因为FastLio里把残差当作观测信息）
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

void PointRGBToXYZI(PointCloudT const *const pi, PointCloudXYZI *const po)
{
    int size = pi->points.size();
    po->resize(size);
    for (int i = 0; i < size; i++)
    {
        po->points[i].x = pi->points[i].x;
        po->points[i].y = pi->points[i].y;
        po->points[i].z = pi->points[i].z;
        po->points[i].intensity = pi->points[i].r;
    }
}

bool poseEstimation(const vector<goMeas> &measurements, cv::Mat &img, Eigen::Matrix3d &K, Eigen::Isometry3d &Tcw)
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimer;
    optimer.setAlgorithm(solver);
    optimer.setVerbose(true);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimer.addVertex(pose);

    int id = 1;

    for (goMeas m : measurements)
    {
        pose_optimize *edge = new pose_optimize();
        edge->set_paras(m.p_lidar, K(0, 0), K(1, 1), K(0, 2), K(1, 2), img);
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        edge->setRobustKernel(rk);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        optimer.addEdge(edge);
    }
    ROS_ERROR("Edges in graph: %lu !!!!!!!!!!!!!!!!!!!!!!!!!", optimer.edges().size());
    optimer.initializeOptimization();
    ROS_ERROR("Edges initialization success!");
    optimer.optimize(opt_num_in_one_frame);
    Tcw = pose->estimate();
    std::cout << "Tcw aft Optimized" << std::endl;
    std::cout << Tcw.matrix() << std::endl;
    optimer.clear();
    return true;
}

// index = 0 refer to frame-frame opt
// index = 0 refer to multi-frame opt,and the pose to be opt is the last elemtne of g2oMeas
void optimize_photometric(deque<Frame *> &opt_buffer)
{
    first_first_point_opt = true;
    // 当程序到第二帧的时候，我们用的是第二帧的雷达位姿
    // 而在此之前，我已经在pubBodyRGB函数中缓存了第二帧投影到第一次Meas的图片的变换
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d body_2_world_1 = Eigen::Isometry3d::Identity();
    body_2_world_1.rotate(state_point.offset_R_L_I.toRotationMatrix());
    body_2_world_1.pretranslate(state_point.offset_T_L_I);
    Eigen::Isometry3d body_2_world_2 = Eigen::Isometry3d::Identity();
    body_2_world_2.rotate(state_point.rot);
    body_2_world_2.pretranslate(state_point.pos);

    Eigen::Isometry3d body_2_world = body_2_world_2 * body_2_world_1;
    Tcw = lidar_cam * lidar_OptLidar * body_2_world.inverse();
    std::cout << "Tcw before Optimized(previous)" << std::endl;
    std::cout << Tcw.matrix() << std::endl;
    // Tcw = multi_opt_buffer.back()->lidar_img;
    Tcw = opt_buffer.back()->pose();
    before_opt_Tcw = Tcw;
    std::cout << "Tcw before Optimized(now)" << std::endl;
    std::cout << Tcw.matrix() << std::endl;

    origTcw = Tcw;
    // 传进这里面的feats_undistort_filter_out都是世界坐标点
    PointCloudXYZI::Ptr laserCloudFullRes(new PointCloudXYZI(opt_buffer.back()->points_map()->size(), 1));
    pcl::copyPointCloud(*(opt_buffer.back()->points_map()), *laserCloudFullRes);
    int size = laserCloudFullRes->points.size();
    if (size == 0)
    {
        ROS_ERROR("No points in opt_stage after filtering process,check Function \"process_pcl_by_depth\" or lidar-cam Translation Matrix");
    }
    // std::vector<g2oMeasure>().swap(opt_meas_buffer);
    opt_meas_buffer.clear();
    opt_meas_buffer.shrink_to_fit();
    Eigen::Vector3d point_world(laserCloudFullRes->points[0].x, laserCloudFullRes->points[0].y, laserCloudFullRes->points[0].z);

    vector<cv::Mat> gray_img_temp;
    for (int i = 0; i < opt_buffer.size(); i++)
    {
        cv::Mat temp_gray;
        cv::cvtColor(opt_buffer[i]->image(), temp_gray, cv::COLOR_BGR2GRAY);
        gray_img_temp.push_back(temp_gray);
    }

    int filter_size = size;
    std::srand((unsigned)time(NULL));
    for (int i = 0; i < filter_size / 10; i++)
    {
        int random_index = std::rand() % (filter_size - 1);
        Eigen::Vector3d point_world(laserCloudFullRes->points[random_index].x, laserCloudFullRes->points[random_index].y, laserCloudFullRes->points[random_index].z);
        for (int j = 0; j < opt_buffer.size() - 1; j++)
        {
            Eigen::Vector3d point_cam = opt_buffer[j]->pose() * point_world;
            // ROS_INFO_STREAM("CamPoint _____" << point_cam);
            if (point_cam(2) == 0)
            {
                ROS_ERROR("Error from divide by ZERO!");
            }
            Eigen::Vector2d pixels = proj3Dto2D(point_cam.x(), point_cam.y(), point_cam.z(), matrixIn_eig(0, 0), matrixIn_eig(1, 1), matrixIn_eig(0, 2), matrixIn_eig(1, 2));
            if (gray_img_temp[j].empty())
            {
                ROS_ERROR("Img Cache is wrong,check img topic!!!");
                continue;
            }
            float gray_temp = getGrayScaleInImg(pixels(0), pixels(1), gray_img_temp[j]);
            if (i == 0 && j == 0)
            {
                std::cout << "in opt stage we porj 1st point to the img of multi_buffer.front()" << std::endl;
                std::cout << pixels(0) << " " << pixels(1) << std::endl;
            }
            if (gray_temp == -1.0)
            {
                continue;
            }
            opt_meas_buffer.push_back(g2oMeasure(point_world, gray_temp, -1.0));
        }
    }
    ROS_WARN("pose estimation launch!");
    // 把最后一个帧的灰度图传进去，作为误差的一项

    // 使用拷贝构造函数深拷贝的对象，ID和内容都一样
    Frame *fe_frame = new Frame(*(opt_buffer.back()));
    FrontEndThread *fe_thread = new FrontEndThread(fe_frame, opt_num_in_one_frame, opt_meas_buffer, matrixIn_eig);
    fe_opt_slider.push_back(fe_thread);

    for (int i = 0; i < gray_img_temp.size(); i++)
    {
        gray_img_temp[i].release();
    }
}

void process_pcl_by_depth(PointCloudXYZI::Ptr pcl_in, PointCloudXYZI::Ptr pcl_out)
{
    // 先转到camera coord再映射到图像上，存三维点映射到二维点的关系，然后在二维点过滤，之后把深度对比一下，那么我觉得可以直接存
    // 但这个用不到后面优化环节，优化环节还是需要原来的三维点，因为你的二维点是投在这一帧上的，所以只能在这过滤了
    int size = pcl_in->points.size();
    std::cout << "Before filter PCL_IN size is " << size << std::endl;
    // 前两个是像素坐标，后一个是该像素的深度值
    vector<Eigen::Vector3d> uv_map;
    PointCloudXYZI::Ptr pcl_in_world(new PointCloudXYZI(size, 1));
    if (size <= 100)
    {
        pcl_out->points.push_back(pcl_in->points.front());
        pcl_in->erase(pcl_in->begin());
        return;
    }
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&pcl_in->points[i], &pcl_in_world->points[i]);
    }
    // 先投到相机坐标系，用这一帧的初始值投，只能这样了
    auto iter_current_frame = frame_map.find(frame_index);
    if (iter_current_frame == frame_map.end())
    {
        ROS_ERROR("Frame Sync Meets Some wrong,Check FrontEnd Code Block!------Horrible Bug!");
        return;
    }
    PointCloudXYZI::iterator sync_pcl_in = pcl_in_world->begin();
    for (auto iter_p = pcl_in->begin(); iter_p != pcl_in->end();)
    {
        Eigen::Vector3d world_p(sync_pcl_in->x, sync_pcl_in->y, sync_pcl_in->z);
        // Eigen::Vector3d temp = multi_opt_buffer.back()->lidar_img * world_p;

        Eigen::Vector3d temp = iter_current_frame->second->pose() * world_p;
        Eigen::Vector2d uv = proj3Dto2D(temp(0), temp(1), temp(2), matrixIn_eig(0, 0), matrixIn_eig(1, 1), matrixIn_eig(0, 2), matrixIn_eig(1, 2));
        if (uv(0) < 0 || uv(0) > iter_current_frame->second->image().cols || uv(1) < 0 || uv(1) > iter_current_frame->second->image().rows)
        {
            iter_p = pcl_in->erase(iter_p);
            sync_pcl_in = pcl_in_world->erase(sync_pcl_in);
        }
        else
        {
            uv_map.push_back(Eigen::Vector3d(uv(0), uv(1), temp(2)));
            iter_p++;
            sync_pcl_in++;
        }
    }
    std::cout << "PCL_IN SIZE BEFORE FILTER " << pcl_in->points.size() << std::endl;
    std::cout << "PCL_IN_WORLD SIZE BEFORE FILTER " << pcl_in_world->points.size() << std::endl;
    std::cout << "UV_MAP SIZE BEFORE FILTER " << uv_map.size() << std::endl;
    while (!pcl_in->points.empty())
    {
        Eigen::Vector3d begin = uv_map.front();
        PointType begin_p = pcl_in->front();
        vector<Eigen::Vector3d>::iterator iter = uv_map.begin();
        PointCloudXYZI::iterator iter_p = pcl_in->begin();
        iter_p = iter_p + 1;
        for (iter = iter + 1; iter != uv_map.end();)
        {
            if (isOverlap(*iter, begin))
            {
                if (iter->z() <= begin.z())
                {
                    // 因为两个iter是同步的，所以当被筛选的uv判定重合时就可以判定其的第三个元素，如果小于被比较的，则替换，然后两个同时删除该iter指向的元素
                    // 然后进行下一个
                    begin = *iter;
                    begin_p = *iter_p;
                    iter = uv_map.erase(iter);
                    iter_p = pcl_in->erase(iter_p);
                }
                else
                {
                    iter = uv_map.erase(iter);
                    iter_p = pcl_in->erase(iter_p);
                }
            }
            else
            {
                iter++;
                iter_p++;
            }
        }
        iter = uv_map.begin();
        iter_p = pcl_in->begin();
        pcl_out->points.push_back(begin_p);
        pcl_in->erase(iter_p);
        uv_map.erase(iter);
    }
    std::cout << "Size after filter" << pcl_out->points.size() << std::endl;
}

void cache_camera_odometry_paras()
{
    cv::Mat src_img = Measures.img;
    // 声明一个空的，给这一帧，在process_filter函数里刷新一下就行了
    PointCloudXYZI::Ptr temp_local_map(new PointCloudXYZI);
    if (initial_frame)
    {
        // 第一帧就用第一次雷达的位姿
        initial_frame = false;
        currentCamerPose.pose.pose.orientation.w = odomAftMapped.pose.pose.orientation.w;
        currentCamerPose.pose.pose.orientation.x = odomAftMapped.pose.pose.orientation.x;
        currentCamerPose.pose.pose.orientation.y = odomAftMapped.pose.pose.orientation.y;
        currentCamerPose.pose.pose.orientation.z = odomAftMapped.pose.pose.orientation.z;
        currentCamerPose.pose.pose.position.x = odomAftMapped.pose.pose.position.x;
        currentCamerPose.pose.pose.position.y = odomAftMapped.pose.pose.position.y;
        currentCamerPose.pose.pose.position.z = odomAftMapped.pose.pose.position.z;
        deltaQ = Quaterniond::Identity();
        deltaPos = V3D(0, 0, 0);
        odomCache_msg = odomAftMapped;
        // last_Opt_cache.img_ref = src_img.clone();
        // 第一帧加入图片,然后tcw不变，在第二帧再重新赋值，因为第一帧计算不出来tcw
        optTcw = Eigen::Isometry3d::Identity();

        Frame *temp_frame_fe = new Frame(temp_local_map, Eigen::Isometry3d::Identity(), src_img.clone());

        frame_map.insert(std::pair<int, Frame *>(temp_frame_fe->ID(), temp_frame_fe));
        ROS_WARN("Frame NO.%d is added into the frame map And Current Frame index in mainThread is %d", temp_frame_fe->ID(), frame_index);
        temp_frame_fe = nullptr;
    }
    else
    {
        double t1 = odomCache_msg.header.stamp.toSec();
        ROS_INFO("last lidar frame time is  :%lf", t1);
        double t2 = odomAftMapped.header.stamp.toSec();
        ROS_INFO("current lidar frame time is  :%lf", t2);
        double t = Measures.img_offset_time + Measures.lidar_beg_time;
        ROS_INFO("current IMG frame time is  :%lf", t);

        currentCamerPose = odomAftInterpolation(t, t1, t2, odomCache_msg, odomAftMapped);
        ROS_INFO("o1: w:%lf,x:%lf,y:%lf,z:%lf || position: x:%lf,y:%lf,z:%lf", odomCache_msg.pose.pose.orientation.w,
                 odomCache_msg.pose.pose.orientation.x, odomCache_msg.pose.pose.orientation.y, odomCache_msg.pose.pose.orientation.z,
                 odomCache_msg.pose.pose.position.x, odomCache_msg.pose.pose.position.y, odomCache_msg.pose.pose.position.z);
        ROS_INFO("lerp: w:%lf,x:%lf,y:%lf,z:%lf || position: x:%lf,y:%lf,z:%lf", currentCamerPose.pose.pose.orientation.w,
                 currentCamerPose.pose.pose.orientation.x, currentCamerPose.pose.pose.orientation.y, currentCamerPose.pose.pose.orientation.z,
                 currentCamerPose.pose.pose.position.x, currentCamerPose.pose.pose.position.y, currentCamerPose.pose.pose.position.z);
        ROS_INFO("o2: w:%lf,x:%lf,y:%lf,z:%lf || position: x:%lf,y:%lf,z:%lf", odomAftMapped.pose.pose.orientation.w,
                 odomAftMapped.pose.pose.orientation.x, odomAftMapped.pose.pose.orientation.y, odomAftMapped.pose.pose.orientation.z,
                 odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z);

        // 计算插值的和当前帧的角度差，就是deltaQ，即 deltaQ*q1 = q2，所以 deltaQ = q2*q1.conjugate
        // 这里转换坐标依然需要对deltaQ取反，因为是 camera->next lidar frame
        Quaterniond q1 = convertRos2Eigen(currentCamerPose);
        q1.normalize();
        Quaterniond q2 = convertRos2Eigen(odomAftMapped);
        q2.normalize();
        deltaQ = q2 * q1.conjugate();
        deltaQ.normalize();
        // 注意，这里计算的是从插值帧的pos移到后一帧点云之间的距离，是插值->点云，所以如果是把插值坐标转到点云坐标系，需要取反
        deltaPos = convertRos2EigenVector(odomCache_msg) - convertRos2EigenVector(currentCamerPose);
        Eigen::Isometry3d temp = Eigen::Isometry3d::Identity();
        temp.rotate(deltaQ.toRotationMatrix());
        temp.translate(deltaPos);
        lidar_OptLidar = temp;
        // world->current body->simulate lidar pos->current cam
        Eigen::Isometry3d frame_trans = lidar_cam * temp * b2w_get().inverse();
        // 平常的帧，在pub body这阶段存的只是该帧的初始值，即用来去优化的
        // cv::Mat temp_gray_img;
        // cv::cvtColor(src_img, temp_gray_img, cv::COLOR_BGR2GRAY);
        // ImgOpt *temp_pointer = new ImgOpt(src_img.clone(), frame_trans);
        Frame *temp_frame_fe = new Frame(temp_local_map, frame_trans, src_img.clone());
        // multi_opt_buffer.push_back(temp_pointer);
        frame_map.insert(std::pair<int, Frame *>(temp_frame_fe->ID(), temp_frame_fe));
        ROS_WARN("Frame NO.%d is added into the frame map And Current Frame index in mainThread is %d", temp_frame_fe->ID(), frame_index);
        // 第一帧就直接用插值，因为用不上优化量，所以直接设置为true，为了后续方便程序运行
        temp_frame_fe->setFrameOptStatus(true);
        // temp_pointer = nullptr;
        temp_frame_fe = nullptr;
        // 对ImgOpt的初始化
        if (times_pub == 1)
        {
            ROS_INFO("initial last opt cam pose");
            Quaterniond q1 = convertRos2Eigen(odomCache_msg);
            q1.normalize();
            std::cout << "cache stage q1: " << q1 << std::endl;
            Quaterniond q2 = convertRos2Eigen(odomAftMapped);
            q2.normalize();
            std::cout << "cache stage q2: " << q1 << std::endl;
            Quaterniond deltaQ_ = q2 * q1.conjugate();
            deltaQ_.normalize();
            std::cout << "(Quaternion)ext's of frame-frame shift-" << std::endl;
            std::cout << deltaQ_ << std::endl;
            V3D deltaP = convertRos2EigenVector(odomCache_msg) - convertRos2EigenVector(odomAftMapped);
            std::cout << "(Translation)ext's of frame-frame shift-" << std::endl;
            std::cout << deltaP << std::endl;
            std::cout << "preframe: " << convertRos2EigenVector(odomCache_msg) << std::endl;
            std::cout << "currentframe: " << convertRos2EigenVector(odomAftMapped) << std::endl;
            Eigen::Isometry3d lidar_initial_lidar = Eigen::Isometry3d::Identity();
            lidar_initial_lidar.rotate(deltaQ_.toRotationMatrix());
            lidar_initial_lidar.pretranslate(deltaP);

            std::cout << "ext's of frame-frame shift" << std::endl;
            std::cout << lidar_initial_lidar.matrix() << std::endl;

            // 这个result相当于是一个body坐标系的雷达点转到前一帧（第一帧）的坐标系然后再结合外参转到摄像机坐标系
            Eigen::Isometry3d result = lidar_cam * lidar_initial_lidar;
            std::cout << "result" << std::endl;
            std::cout << result.matrix() << std::endl;
            // last_Opt_cache.lidar_img = result;
            // 此时在第二帧，把第一帧的值赋给multi_opt_buffer
            // 要注意，给第一帧的也得是world->cam的变换，所以这个得统一，先world->body，再body->cam

            // 这里存的转换body_2_world其实是世界坐标的一个点转到第二帧雷达自身坐标系
            // 到第二帧，第一帧的参数才真正设置完，在之后的优化中就可以使用第一帧的lidar_img了
            // multi_opt_buffer[0]->lidar_img = result * b2w_get().inverse();
            frame_map.find(frame_map.begin()->first)->second->setPose(result * b2w_get().inverse());
        }
        odomCache_msg = odomAftMapped;
    }
}

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<cv::DMatch> matches,
                          cv::Mat &R, cv::Mat &t)
{
    // 相机内参,TUM Freiburg2
    cv::Mat K = matrix_in;

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // //-- 计算基础矩阵
    // cv::Mat fundamental_matrix;
    // fundamental_matrix = cv::findFundamentalMat ( points1, points2, cv::FM_8POINT );
    // cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    cv::Point2d principal_point(matrixIn_eig(0, 2), matrixIn_eig(1, 2));
    double focal_length = matrixIn_eig(1, 1);
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl
         << essential_matrix << endl;

    // //-- 计算单应矩阵
    // cv::Mat homography_matrix;
    // homography_matrix = cv::findHomography ( points1, points2, cv::RANSAC, 3 );
    // cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R is " << std::endl
              << R << std::endl;
    std::cout << "t is " << std::endl
              << t << std::endl;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches)
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<cv::DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}
std::unordered_map<int, vector<goMeas>> meas_map;
// std::unordered_map<int,vector<float>> gray_map_backend;
int backend_opt_num = 1;

void pose_opt_backend(const deque<Frame *> backend_buffers, const ros::Publisher &odom_pub, const ros::Publisher &path_pub)
{
    meas_map.clear();
    ROS_WARN("Enter");
    PointCloudXYZI::Ptr laserCloudFullRes(new PointCloudXYZI());
    for (int i = 0; i < backend_buffers.size(); i++)
    {
        *laserCloudFullRes += *(backend_buffers[i]->points_map());
    }
    int size = laserCloudFullRes->points.size();
    if (size == 0)
    {
        ROS_ERROR("No points in opt_stage after filtering process,check Function \"process_pcl_by_depth\" or lidar-cam Translation Matrix");
    }
    // 因为下面的k的for循环是根据所有的优化帧的顺序选取的，而其中有跳跃的点在于k==i这个条件
    // 所以我在这里按照backend_buffer里的顺序弄出一个灰度图的buffer，就可以产生观测信息，并在结束的时候释放掉这个vector，这样就不需要在结构体里再存储一个额外的灰度图了
    std::vector<cv::Mat> gray_img_temp;
    gray_img_temp.reserve(backend_buffers.size());
    for (int i = 0; i < backend_buffers.size(); i++)
    {
        cv::Mat temp_gray;
        if (!backend_buffers[i]->image().empty())
        {
            cv::cvtColor(backend_buffers[i]->image(), temp_gray, cv::COLOR_BGR2GRAY);
            gray_img_temp.push_back(temp_gray);
        }
        else
        {
            ROS_ERROR("Image Cache has something wrong,CHECK IT!");
        }
        temp_gray.release();
    }
    int filter_size = size;
    ROS_ERROR("cloud size %d", filter_size);
    std::srand((unsigned)time(NULL));
    for (int i = 0; i < backend_opt_pose_num; i++)
    {
        std::vector<goMeas> temp;
        for (int j = 0; j < filter_size / 150; j++)
        {
            int random_index = std::rand() % (filter_size - 1);
            Eigen::Vector3d point_world(laserCloudFullRes->points[random_index].x, laserCloudFullRes->points[random_index].y, laserCloudFullRes->points[random_index].z);
            float weight = 0;
            vector<goMeas> temp_meas_buffer;
            for (int k = 0; k < backend_opt_pose_num; k++)
            {
                if (k == i)
                {
                    continue;
                }
                Eigen::Vector3d point_cam = backend_buffers[k]->pose() * point_world;
                Eigen::Vector2d pixels = proj3Dto2D(point_cam.x(), point_cam.y(), point_cam.z(), matrixIn_eig(0, 0), matrixIn_eig(1, 1), matrixIn_eig(0, 2), matrixIn_eig(1, 2));
                float gray_temp = getGrayScaleInImg(pixels(0), pixels(1), gray_img_temp[k]);

                if (gray_temp == -1.0)
                {
                    continue;
                }
                // 以点是否与优化帧共视作为权重
                weight += 1.0;
                temp_meas_buffer.push_back(goMeas(point_world, gray_temp, -1.0));
            }
            int size_tmb = temp_meas_buffer.size();
            if (size_tmb == 0)
            {
                continue;
            }
            Eigen::VectorXd gray_array(size_tmb);
            for (int k = 0; k < size_tmb; k++)
            {
                gray_array(k) = temp_meas_buffer[k].grayscale / 255.0f;
            }
            double mean_gray_array = gray_array.mean();
            double variance = (gray_array.array() - mean_gray_array).square().sum() / (double)size_tmb;
            double std_deviation = std::sqrt(variance);
            double point_weight = weight / std::exp(std_deviation);
            std::vector<double> co_array(size_tmb);
            for (int k = 0; k < size_tmb; k++)
            {
                temp_meas_buffer[k].weight = point_weight;
                temp.push_back(temp_meas_buffer[k]);
                co_array[k] = temp_meas_buffer[k].grayscale;
            }
            weight_point.push_back(poWeightL(point_weight, weight, variance, co_array));
        }
        // 第0个就是第0帧需要的观测数据
        ROS_WARN("Push %d-th frame mease in buffer,continue! --------%lu", i, temp.size());
        meas_map.insert(std::pair<int, vector<goMeas>>(i, temp));
    }
    // 这个变量原本的意义到这里已经用不上了，所以直接赋值，重复利用
    filter_size = backend_buffers.size();
    // 为了多线程，必须让该对象里面的所有用于优化的数据都是自含的，不然主线程很有可能发生内存泄漏
    vector<Frame *> pose_opt_copy;
    for (int i = 0; i < filter_size; i++)
    {
        // 为了保持同一性，ImgOpt里面就应该存彩色图，转换的话放到线程里去转换吧，反正也就是20张转换
        Frame *temp_be_frame = new Frame(*backend_buffers[i]);
        pose_opt_copy.push_back(temp_be_frame);
    }
    BackendThread *be_pose_opt = new BackendThread(pose_opt_copy, opt_num_in_one_frame, meas_map, matrixIn_eig);
    backend_opt_slider.push_back(be_pose_opt);
    backend_opt_num++;
    for (int i = 0; i < gray_img_temp.size(); i++)
    {
        gray_img_temp[i].release();
    }
}

void release_vector_with_pointer(vector<ImgOpt *> &buffer)
{
    int size = buffer.size();
    for (int i = 0; i < size; i++)
    {
        delete buffer[i];
        buffer[i] = nullptr;
    }
    buffer.clear();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("filter_size_camera_opt", filter_size_camera_opt, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<int>("common/opt_frame_fn", opt_frame_fn, 2);
    nh.param<int>("common/opt_num_in_one_frame", opt_num_in_one_frame, 20);
    nh.param<int>("common/backend_opt_pose_num", backend_opt_pose_num, 10);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("mapping/cam_lidar_ext", lidar_cam_ext, vector<double>());
    nh.param<vector<double>>("mapping/cam_intrinsic", cam_intrinsic, vector<double>());

    nh.param<int>("publish/frame_num_to_stop", frame_num_to_stop, 20);
    nh.param<int>("publish/frame_map_to_save", frame_map_to_save, 20);
    nh.param<int>("publish/rgb_point_save_num", rgb_point_save_num, 20);
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    path_camera.header.stamp = ros::Time::now();
    path_camera.header.frame_id = "camera_init";

    std::time_t currentTime = std::time(nullptr);
    dateString = timestampToDateString(currentTime);

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    /*
    0.0771635 -0.996892 0.0158657 -0.175356
    0.130788 -0.00565472 -0.991394 -0.243168
    0.988403 0.0785745 0.129945 -0.178752
    0 0 0 1
    */
    /*
    -0.040507,-0.998998,-0.019029,0.163793
    0.0388064,0.0174573,-0.999094,0.0762877
    0.998425,-0.0412087,0.0380603,-0.438607
    0,0,0,1
    */
    //---------camera parameters-----------//
    // std::vector<double> intrinsic = {817.846, 0, 1000.37, 0, 835.075, 551.246, 0, 0, 1};
    std::vector<double> distortion = {-0.2658, 0.0630, 0.0026, 0.0015, 0};
    // std::vector<double> extrinsic = {-0.040507, -0.998998, -0.019029, 0.163793, 0.0388064, 0.0174573, -0.999094, 0.0762877, 0.998425, -0.0412087, 0.0380603, -0.438607, 0, 0, 0, 1};
    // std::vector<double> extrinsic = {0.0771635, -0.996892, 0.0158657, -0.175356,
    //                                  0.130788, -0.00565472, -0.991394, -0.243168,
    //                                  0.988403, 0.0785742, 0.129945, -0.178752, 0, 0, 0, 1};

    double matrix1[3][3] = {{cam_intrinsic[0], cam_intrinsic[1], cam_intrinsic[2]}, {cam_intrinsic[3], cam_intrinsic[4], cam_intrinsic[5]}, {cam_intrinsic[6], cam_intrinsic[7], cam_intrinsic[8]}};
    double matrix2[3][4] = {{lidar_cam_ext[0], lidar_cam_ext[1], lidar_cam_ext[2], lidar_cam_ext[3]}, {lidar_cam_ext[4], lidar_cam_ext[5], lidar_cam_ext[6], lidar_cam_ext[7]}, {lidar_cam_ext[8], lidar_cam_ext[9], lidar_cam_ext[10], lidar_cam_ext[11]}};

    matrix_in = cv::Mat(3, 3, CV_64F, matrix1);
    matrix_out = cv::Mat(3, 4, CV_64F, matrix2);

    cv::cv2eigen(matrix_in, matrixIn_eig);
    cv::cv2eigen(matrix_out, matrixOut_eig);

    /*lidar_cam isometry3d matrix initialization*/
    Eigen::Matrix3d rotation_part = matrixOut_eig.block<3, 3>(0, 0);
    Eigen::Vector3d translation_part = matrixOut_eig.block<3, 1>(0, 3);
    lidar_cam.rotate(rotation_part);
    lidar_cam.pretranslate(translation_part);
    // set intrinsic parameters of the camera
    // camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    // camera_matrix.at<double>(0, 0) = intrinsic[0];
    // camera_matrix.at<double>(0, 2) = intrinsic[2];
    // camera_matrix.at<double>(1, 1) = intrinsic[4];
    // camera_matrix.at<double>(1, 2) = intrinsic[5];

    // // set radial distortion and tangential distortion
    // distortion_coef = cv::Mat::zeros(5, 1, CV_64F);
    // distortion_coef.at<double>(0, 0) = distortion[0];
    // distortion_coef.at<double>(1, 0) = distortion[1];
    // distortion_coef.at<double>(2, 0) = distortion[2];
    // distortion_coef.at<double>(3, 0) = distortion[3];
    // distortion_coef.at<double>(4, 0) = distortion[4];

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    // 在这里初始化，然后传进去 ”h_share_model“ 这个函数给函数指针h_dyn_share
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    // cache 1000 msgs in one pool

    ros::Subscriber sub_img = nh.subscribe("/camera/image", 100000, img_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudFull_body_rgb = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body_rgb", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubOdomAftMapped_cam = nh.advertise<nav_msgs::Odometry>("/Odometry_cam", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);
    ros::Publisher pubPath_camera = nh.advertise<nav_msgs::Path>("/path_camera", 100000);
    ros::Publisher pubLaserCloudRGBFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_rgb", 100000);
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    int index_stop = 0;
    bool once = true;
    unsigned long diff;
    whole_time_one_frame_start = omp_get_wtime();
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, nullptr);

    local_map_pc->height = 1;
    local_map_pc->is_dense = true;
    while (status)
    {
        if (flg_exit)
            break;
        // 调用该函数后可以继续执行下面的代码
        ros::spinOnce();
        if (sync_packages_img(Measures))
        {

            if (flg_first_scan)
            {
                ROS_INFO("First flg work!");
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            ROS_INFO("after sync,points size is :%lu", Measures.lidar->points.size());
            // after feats_undistort pass through the Process function,it contains all the pcl with respected to the i-th scan lidar frame(10 HZ)
            p_imu->Process(Measures, kf, feats_undistort);
            ROS_INFO("after process,points size is :%lu", feats_undistort->points.size());

            state_point = kf.get_x();
            // IMU坐标系下的lidar位置
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            // 这个存的是优化之前的位置信息
            if (feats_undistort->empty())
            {
                ROS_WARN("EMPTY UNDISTORT!\n");
            }
            if (feats_undistort == nullptr)
            {
                ROS_WARN("NULLPTR UNDISTORT!\n");
            }
            if (feats_undistort->empty() || (feats_undistort == nullptr))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            // 用VoxelGrid filter进行下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            // downSizeFilterSurf.setInputCloud(cloudRGB);
            // 在这里为feats_down_body赋值，即存入下采样的特征点的自身坐标系下的坐标
            // downSizeFilterSurf.filter(*cloudRGB_down_body);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            // feats_down_size = cloudRGB_down_body->points.size();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        // pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                ROS_WARN("BUILDING IKD-Tree!");
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            // looks like extrinsic params,
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (0) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            // 在 “update_iterated_dyn_share_modified”里更新了 x_，所以用getx直接获得矫正好的状态
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            // 经过后端优化迭代完的位姿，再来计算一下lidar在IMU坐标系下的位置
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            // 在这一步会把geoQuat进行set，即给地图点一个方向
            publish_odometry(pubOdomAftMapped);
            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            /******* Publish points *******/
            if (path_en)
            {
                publish_path(pubPath);
                // publish_path_interpolate(pubPath_camera);
            }

            if (scan_pub_en || pcd_save_en)
            {
                // publish_frame_world_rgb(pubLaserCloudRGBFull);
                publish_frame_world(pubLaserCloudFull);
            }

            // TODO
            // optimized function in here

            if (scan_pub_en && scan_body_pub_en)
            {
                publish_frame_body(pubLaserCloudFull_body);
                // publish_frame_body_rgb(pubLaserCloudFull_body_rgb);
            }
            // 该函数把优化过程比如插值或者要缓存的参数都包括在内，每帧运行一次

            cache_camera_odometry_paras();
            feats_undistort_filter->clear();
            pcl::copyPointCloud(*feats_undistort, *feats_undistort_filter);
            ROS_ERROR("ERROR--------feats_undistort_filter_out SIZE %d In Frame Map", feats_undistort_filter->size());
            PointCloudXYZI::Ptr feats_undistort_filter_out(new PointCloudXYZI);
            // TODO 这里明明可以处理出来就是world点，不需要再转换了，以后做
            if (frame_index > 0)
            {
                feats_undistort_filter_out->clear();
                feats_undistort_filter_out->height = 1;
                feats_undistort_filter_out->is_dense = true;
                process_pcl_by_depth(feats_undistort_filter, feats_undistort_filter_out);
            }
            else
            {
                feats_undistort_filter_out = feats_undistort_filter;
            }
            int feats_undistort_filter_size = feats_undistort_filter_out->size();
            PointCloudXYZI::Ptr feats_undistort_filter_out_world(new PointCloudXYZI(feats_undistort_filter_size, 1));

            ROS_ERROR("ERROR--------feats_undistort_filter_out SIZE %d In Frame Map", feats_undistort_filter_out->size());
            ROS_ERROR("ERROR--------feats_undistort_filter_out SIZE %d In Frame Map", feats_undistort_filter_out_world->size());
            for (int i = 0; i < feats_undistort_filter_size; i++)
            {
                pointBodyToWorld(&(feats_undistort_filter_out->points[i]), &(feats_undistort_filter_out_world->points[i]));
                // pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ROS_ERROR("ERROR--------feats_undistort_filter_out SIZE %d In Frame Map", feats_undistort_filter_out->size());
            auto it = frame_map.find(frame_index);
            if (it != frame_map.end())
            {
                pcl::copyPointCloud(*feats_undistort_filter_out_world, *(it->second->points_map()));
                ROS_ERROR("My cache Frame's points size is %lu", it->second->points_map()->size());
            }
            else
            {
                ROS_ERROR("ERROR--------MainThread: No corresponding frame with ID = %d In Frame Map", frame_index);
                return 0;
            }
            feats_undistort_filter->clear();
            feats_undistort_filter->height = 1;
            feats_undistort_filter->is_dense = true;
            feats_undistort_filter_out_world->clear();
            feats_undistort_filter_out_world.reset();

            // 声明成临时变量了，因为多线程情况一个while循环的时间执行不了20帧的后端优化
            int fe_index = 0;
            int orig_fe_opt_slider = fe_opt_slider.size();
            vector<FrontEndThread *>::iterator frame_iter = fe_opt_slider.begin();
            while (frame_iter != fe_opt_slider.end())
            {
                if ((*frame_iter)->is_finished())
                {
                    Frame *temp = frame_map.at((*frame_iter)->get_opt_frame()->ID());
                    temp->setPose((*frame_iter)->get_opt_pose());
                    temp->setFrameOptStatus(true);
                    if (backend_opt_buffer.size() < backend_opt_pose_num)
                    {
                        backend_opt_buffer.push_back(temp);
                    }
                    // 这里delete无所谓，因为FrontEndThread的frame是自含的，不会影响主线程的内容
                    delete (*frame_iter);
                    (*frame_iter) = nullptr;
                    frame_iter = fe_opt_slider.erase(frame_iter);
                    fe_index++;
                }
                else
                {
                    frame_iter++;
                }
            }
            ROS_WARN("FrontEnd----------- %d/%d is Finished!", fe_index, orig_fe_opt_slider);
            opt_frame_index_buffer.push_back(frame_index);
            frame_index++;

            for (auto it = opt_frame_index_buffer.begin(); it != opt_frame_index_buffer.end();)
            {
                auto iter_start = frame_map.find(*it);
                if (iter_start != frame_map.end())
                {
                    frame_opt_buffer.clear();
                    // 按照顺序的话，我们优化的是最新的帧，所以该帧一定未优化过
                    bool last_frame = true;
                    for (auto rit = std::map<int, Frame *>::reverse_iterator(iter_start); rit != frame_map.rend(); ++rit)
                    {
                        if (frame_opt_buffer.size() >= opt_frame_fn)
                        {
                            break;
                        }
                        if (rit->second->OptStatus())
                        {
                            frame_opt_buffer.push_front(rit->second);
                        }
                    }
                    // 当运行帧大于优化的帧数时，我们需要让滑动窗口里的观测帧满足条件才能优化，不然就不优化这帧，而等到下次条件够了再优化
                    if (frame_index > opt_frame_fn)
                    {
                        if (frame_opt_buffer.size() < opt_frame_fn)
                        {
                            ++it;
                            continue;
                        }
                        else
                        {
                            optimize_photometric(frame_opt_buffer);
                            it = opt_frame_index_buffer.erase(it);
                            ROS_WARN("frame_opt_buffer size is %lu", frame_opt_buffer.size());
                        }
                    }
                    else
                    {
                        if (frame_opt_buffer.size() > 1)
                        {
                            optimize_photometric(frame_opt_buffer);
                            it = opt_frame_index_buffer.erase(it);
                            ROS_WARN("frame_opt_buffer size is %lu", frame_opt_buffer.size());
                        }
                        else
                        {
                            ++it;
                            continue;
                        }
                    }
                }
                else
                {
                    ++it;
                }
            }
            ROS_ERROR("2222222222222222222 %d In Frame Map", feats_undistort_filter_out->size());
            if (backend_opt_buffer.size() >= backend_opt_pose_num)
            {
                ROS_WARN("Backend Opt Start!");
                pose_opt_backend(backend_opt_buffer, pubOdomAftMapped_cam, pubPath_camera);
                backend_opt_buffer.clear();
            }

            vector<BackendThread *>::iterator iter = backend_opt_slider.begin();
            int indices = 0;
            int orig_size = backend_opt_slider.size();
            // Pub阶段
            while (iter != backend_opt_slider.end())
            {
                if ((*iter)->is_finished())
                {
                    ROS_WARN("one is finished!!!");
                    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> cache_pose = (*iter)->get_opt_pose();
                    int size = cache_pose.size();
                    // for (int j = 0; j < size; j++)
                    // {
                    //     publish_odometry_camera_backend(pubOdomAftMapped_cam, cache_pose[j]);
                    //     publish_path_interpolate_backend(pubPath_camera, cache_pose[j]);
                    // }
                    // 这是后端的某个线程返回的Frame buffer
                    std::vector<Frame *> temp_opt_buffer = (*iter)->get_opt_buffer();
                    // 我们需要把它同步回主线程
                    std::vector<Frame *> main_thread_be_opt_buffer;
                    for (int i = 0; i < temp_opt_buffer.size(); i++)
                    {
                        auto it = frame_map.find(temp_opt_buffer[i]->ID());
                        if (it != frame_map.end())
                        {
                            Frame *temp = it->second;
                            main_thread_be_opt_buffer.push_back(temp);
                        }
                        else
                        {
                            continue;
                        }
                    }
                    PointCloudXYZI::Ptr localMap_ptr = (*iter)->get_opt_map();
                    ROS_WARN("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!This backend opt object has %lu points", localMap_ptr->size());
                    publish_frame_world_rgb(pubLaserCloudRGBFull, main_thread_be_opt_buffer, localMap_ptr);
                    ROS_WARN("!!!!!!!!!!!After Publish This backend opt object has %lu points", localMap_ptr->size());
                    delete (*iter);
                    iter = backend_opt_slider.erase(iter);
                }
                else
                {
                    iter++;
                }
                indices++;
            }
            ROS_WARN("Backend-----------%d / %d is finished!!!", indices, orig_size);
            // 清除操作，把前端优化过的，以及Publish过的都删了，只有满足这两个条件的才会删除
            auto iter_map = frame_map.begin();
            while (iter_map != frame_map.end())
            {
                if (iter_map->second->OptStatus())
                {
                    if (iter_map->second->published())
                    {
                        delete iter_map->second;
                        iter_map = frame_map.erase(iter_map);
                    }
                    else
                    {
                        iter_map++;
                    }
                }
                else
                {
                    iter_map++;
                }
            }

            // publish_odometry_camera(pubOdomAftMapped_cam);
            // publish_frame_world_rgb(pubLaserCloudRGBFull);

            // publish_path_interpolate(pubPath_camera);

            // geometry_msgs::Quaternion temp_q = odomAftMapped.pose.pose.orientation;
            // geometry_msgs::Point temp_t = odomAftMapped.pose.pose.position;
            // Eigen::Quaterniond ei_q;
            // Eigen::Vector3d ei_t;
            // Eigen::Quaterniond opt_ei_q;
            // Eigen::Vector3d opt_ei_t;
            // getOriginalCoordByTransMatrix(opt_ei_q, opt_ei_t, multi_opt_buffer.back()->lidar_img);
            // getOriginalCoordByTransMatrix(ei_q, ei_t, before_opt_Tcw);
            // LogVariable cur_frame_log = LogVariable(Measures.lidar_end_time, cloudRGBTemp->points.size(), invalid_count, valid_orig_count, optTcw, origTcw,
            //                                         ei_q, ei_t, opt_ei_q, opt_ei_t, after_opt_valid_ratio, before_opt_valid_ratio);
            // cache_all_log.push_back(cur_frame_log);
            // //
            //

            index_stop++;
            if (index_stop >= frame_num_to_stop)
            {
                ROS_INFO("accumlated num is acheiving stop time");
                break;
            }
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    while (1)
    {
        vector<BackendThread *>::iterator iter = backend_opt_slider.begin();
        while (iter != backend_opt_slider.end())
        {
            if (!(*iter)->is_finished())
            {
                iter++;
            }
            else
            {
                ROS_WARN("PUB!");
                vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> cache_pose = (*iter)->get_opt_pose();
                int size = cache_pose.size();
                for (int j = 0; j < size; j++)
                {
                    publish_odometry_camera_backend(pubOdomAftMapped_cam, cache_pose[j]);
                    publish_path_interpolate_backend(pubPath_camera, cache_pose[j]);
                }
                delete (*iter);
                (*iter) = nullptr;
                iter = backend_opt_slider.erase(iter);
            }
        }
        if (backend_opt_slider.size() == 0)
        {
            ROS_WARN("ALL POSES HAVE BEEN OPT!");
            break;
        }
    }
    gettimeofday(&tv_end, nullptr);
    diff = 1000000 * (tv_end.tv_sec - tv_start.tv_sec) + tv_end.tv_usec - tv_start.tv_usec;
    whole_time_one_frame = omp_get_wtime();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    // if (pcl_wait_save->size() > 0 && pcd_save_en)
    // {
    //     string file_name = string("scans.pcd");
    //     string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
    //     pcl::PCDWriter pcd_writer;
    //     cout << "current scan saved to /PCD/" << file_name<<endl;
    //     pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    // }

    // ColorForMap();
    if (pcl_wait_save_rgb->size() > 0 && pcd_save_en)
    {

        string file_name = string("scans_rgb_" + dateString + ".pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_rgb);
    }

    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        std::time_t currentTime = std::time(nullptr);
        std::string dateString = timestampToDateString(currentTime);
        string file_name = string("scans_" + dateString + ".pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }
    FILE *opt_log_file;
    double cur_timestamp = omp_get_wtime();
    std::stringstream ss;
    ss << std::setprecision(10) << cur_timestamp;
    string opt_log_dir = root_dir + "/Log/m_opt_log_" + ss.str() + ".txt";
    ss.clear();
    opt_log_file = fopen(opt_log_dir.c_str(), "a");
    fprintf(opt_log_file, "true time \t\t frame end time \t\t N \t opt ratio \t raw ratio \t\t opt rot \t\t opt trans \t\t rot \t\t trans \n");
    for (int i = 0; i < cache_all_log.size(); i++)
    {
        LogVariable temp = cache_all_log[i];
        Quaterniond opt_q(temp.optTcw.rotation());
        // Eigen::Vector3d opt_trans = temp.optTcw.translation();
        Eigen::Vector3d opt_trans = temp.opt_trans;
        Quaterniond q(temp.origTcw.rotation());
        // Eigen::Vector3d t = temp.origTcw.translation();
        Eigen::Vector3d t = temp.trans;
        V3D opt_euler_rot = QuaternionToEuler(temp.opt_rot);
        V3D ori_euler_rot = QuaternionToEuler(temp.rot);

        // fprintf(opt_log_file, "%0.8f|%d|%d|%d|%f,%f,%f,%f|%f,%f,%f|%f,%f,%f,%f|%f,%f,%f|\n", temp.frame_time, temp.sum_point_size, temp.opt_uv_size,
        //         temp.valid_size, opt_q.w(), opt_q.x(), opt_q.y(), opt_q.z(), opt_trans(0), opt_trans(1), opt_trans(2), q.w(), q.x(), q.y(), q.z(), t(0), t(1), t(2));

        fprintf(opt_log_file, "%0.8f|%0.8f|%d \t |%f% \t |%f% \t |%f,%f,%f|%f,%f,%f|%f,%f,%f|%f,%f,%f| \n", tv_end.tv_sec - tv_start.tv_sec, temp.frame_time, temp.sum_point_size, temp.opt_ratio,
                temp.ori_ratio, opt_euler_rot.x(), opt_euler_rot.y(), opt_euler_rot.z(), opt_trans(0), opt_trans(1), opt_trans(2), ori_euler_rot.x(), ori_euler_rot.y(), ori_euler_rot.z(), t(0), t(1), t(2));
    }
    fclose(opt_log_file);
    cout << "current opt vars log saved to" << opt_log_dir << endl;
    // return 0;

    FILE *weight_log_file;
    double cur_timestamp_weight = omp_get_wtime();
    std::stringstream ss_weight;
    ss_weight << std::setprecision(10) << cur_timestamp_weight;
    string weight_log_dir = root_dir + "/Log/all_points_weight_log_" + ss_weight.str() + ".txt";
    ss_weight.clear();
    weight_log_file = fopen(weight_log_dir.c_str(), "a");
    fprintf(weight_log_file, "id \t weight \t co_visible \t variacne \t\t gray_scale \n");
    for (int i = 0; i < weight_point.size(); i++)
    {
        auto temp = weight_point[i];
        // fprintf(opt_log_file, "%0.8f|%d|%d|%d|%f,%f,%f,%f|%f,%f,%f|%f,%f,%f,%f|%f,%f,%f|\n", temp.frame_time, temp.sum_point_size, temp.opt_uv_size,
        //         temp.valid_size, opt_q.w(), opt_q.x(), opt_q.y(), opt_q.z(), opt_trans(0), opt_trans(1), opt_trans(2), q.w(), q.x(), q.y(), q.z(), t(0), t(1), t(2));

        fprintf(weight_log_file, "%d \t|%0.4f\t| %d\t|%.5f|", i, temp.weight, temp.co_visible_num, temp.variance);
        for (int k = 0; k < temp.co_gray_array.size(); k++)
        {
            fprintf(weight_log_file, "[%f]", temp.co_gray_array[k]);
        }
        fprintf(weight_log_file, "\n");
    }
    fclose(weight_log_file);
    cout << "current weight log saved to" << weight_log_dir << endl;

    return 0;
}
