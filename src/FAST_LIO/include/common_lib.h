#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <fast_lio/Pose6D.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

#define USE_IKFOM

#define PI_M (3.14159265358)
#define G_m_s2 (9.81)   // Gravaty const in GuangDong/China
#define DIM_STATE (18)  // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12) // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN (6.0)
#define LIDAR_SP_LEN (2)
#define INIT_COV (1)
#define NUM_MATCH_POINTS (5)
#define MAX_MEAS_DIM (10000)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define LIDAR_CAM_VEC_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],V[9],V[10],V[11],V[12],V[13],V[14],V[15]
#define CONSTRAIN(v, min, max) ((v > min) ? ((v < max) ? v : max) : min)
#define ARRAY_FROM_EIGEN(mat) mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat) vector<decltype(mat)::Scalar>(mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "Log/" + name))

typedef fast_lio::Pose6D Pose6D;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef pcl::PointXYZRGB PointTypeRGB;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

#define MD(a, b) Matrix<double, (a), (b)>
#define VD(a) Matrix<double, (a), 1>
#define MF(a, b) Matrix<float, (a), (b)>
#define VF(a) Matrix<float, (a), 1>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

struct LogVariable
{

    LogVariable(double frame_time, int sum_point_size, int opt_uv_size, int valid_size, Eigen::Isometry3d optTcw, Eigen::Isometry3d origTcw,
                Eigen::Quaterniond rot, Eigen::Vector3d trans, Eigen::Quaterniond opt_rot, Eigen::Vector3d opt_trans,float opt_ratio,float ori_ratio)
        : opt_uv_size(opt_uv_size), optTcw(optTcw), origTcw(origTcw), valid_size(valid_size), sum_point_size(sum_point_size), frame_time(frame_time),
          rot(rot), trans(trans), opt_rot(opt_rot), opt_trans(opt_trans),opt_ratio(opt_ratio),ori_ratio(ori_ratio){};
    // 当前帧的时间，以雷达扫描帧结束为记录点
    double frame_time;
    // 当前帧的所有点数
    int sum_point_size;
    // 表示优化过的投影到当前帧的有效像素大小
    int opt_uv_size;
    // 当前未优化过的有效的投到图像帧的点数
    int valid_size;
    // 优化过的当前帧转换关系，即将世界点投到当前图片帧对应的相机坐标系
    Eigen::Isometry3d optTcw;
    // 未优化的，即我们通过将点从world->body->delta->extrinsic计算出来的
    Eigen::Isometry3d origTcw;
    // 未优化前的相机相对IMU（第一个），即世界坐标的旋转
    Eigen::Quaterniond rot;
    // 未优化前的相机相对IMU（第一个），即世界坐标的平移
    Eigen::Vector3d trans;
    // 优化前的相机相对IMU（第一个），即世界坐标的旋转
    Eigen::Quaterniond opt_rot;
    // 优化前的相机相对IMU（第一个），即世界坐标的平移
    Eigen::Vector3d opt_trans;
    //优化后投影的无效点比例
    float opt_ratio;
    //优化前投影的无效点比例
    float ori_ratio;
};

typedef struct g2oMeasure
{
    g2oMeasure(Eigen::Vector3d p, float gray, float weight) : p_lidar(p), grayscale(gray), weight(weight){};
    g2oMeasure()
    {
        p_lidar = Eigen::Vector3d::Zero();
        grayscale = -1.0;
        weight = -1.0;
    };
    Eigen::Vector3d p_lidar;
    float grayscale;
    float weight;
} goMeas;

// 不能用指针，因为每一帧我们存储的图像，都会因为Measures的改变而改变，所以图像必须用复制
// g2o那一块可以用指针，因为那个边只会存在一帧，到下一帧就是新的边了
typedef struct ImgOptimized
{
    ImgOptimized()
    {
        img_ref = cv::Mat();
        lidar_img = Eigen::Isometry3d::Identity();
    };
    ImgOptimized(cv::Mat img, Eigen::Isometry3d transform) : img_ref(img), lidar_img(transform){};
    cv::Mat img_ref;
    // 这里保存的是如何把你用于优化的雷达那一帧坐标系下的点投影到该帧图像上，所以如果你换了其他雷达帧，就得先转到该雷达帧坐标系下再使用
    // 我现在先把雷达点转到世界坐标，这样优化的就是世界坐标如何转到相机坐标系的总的R了，之后的投影就直接用这个来做
    Eigen::Isometry3d lidar_img;
    // //当前帧雷达坐标系在世界坐标的位置和方向
    // Eigen::Vector3d lidar_pos_w;
    // Quaterniond lidar_ori_w;

} ImgOpt;


typedef struct Pcl_Set{
    Pcl_Set(PointType point):point(point){};
    PointType point;

    bool operator <(const Pcl_Set& r_point) const{
        //只有当被比较的点（要插入的点）不仅在这个范围内且
        //当该点与红黑树中的点比较距离大于半径，说明两点不相交，此时再判断深度哪个小，如果该点小则留下该点
        //这里不用三维的原因是深度不应该作为比较的标准，比如有个表面凹凸不平的物体，自然其在y-z是同一个点但是加了x的距离可能就不满足相等的标准了
        if(sqrt(powl(this->point.y-r_point.point.y,2)+powl(this->point.z-r_point.point.z,2))<0.05f){
            if(this->point.x<r_point.point.x){
                return false;
            }
        }
        return (sqrt(powl(this->point.x-r_point.point.x,2)+powl(this->point.y-r_point.point.y,2)+powl(this->point.z-r_point.point.z,2))>=0.05f&&\
        this->point.x<r_point.point.x);
    }
} Pset;


struct MeasureGroup // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        img_offset_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
        is_lidar_end = false;
        lidar_scan_index_now = 0;
        last_update_time = 0.0;
    };
    double lidar_beg_time;
    double lidar_end_time;
    double img_offset_time;
    bool is_lidar_end;
    int lidar_scan_index_now;
    double last_update_time;
    PointCloudXYZI::Ptr lidar;
    deque<sensor_msgs::Imu::ConstPtr> imu;
    cv::Mat img;
};

struct StatesGroup
{
    StatesGroup()
    {
        this->rot_end = M3D::Identity();
        this->pos_end = Zero3d;
        this->vel_end = Zero3d;
        this->bias_g = Zero3d;
        this->bias_a = Zero3d;
        this->gravity = Zero3d;
        this->cov = MD(DIM_STATE, DIM_STATE)::Identity() * INIT_COV;
        this->cov.block<9, 9>(9, 9) = MD(9, 9)::Identity() * 0.00001;
    };

    StatesGroup(const StatesGroup &b)
    {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
    };

    StatesGroup &operator=(const StatesGroup &b)
    {
        this->rot_end = b.rot_end;
        this->pos_end = b.pos_end;
        this->vel_end = b.vel_end;
        this->bias_g = b.bias_g;
        this->bias_a = b.bias_a;
        this->gravity = b.gravity;
        this->cov = b.cov;
        return *this;
    };

    StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add)
    {
        StatesGroup a;
        a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
        a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
        a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
        a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
        a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
        a.cov = this->cov;
        return a;
    };

    StatesGroup &operator+=(const Matrix<double, DIM_STATE, 1> &state_add)
    {
        this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
        this->pos_end += state_add.block<3, 1>(3, 0);
        this->vel_end += state_add.block<3, 1>(6, 0);
        this->bias_g += state_add.block<3, 1>(9, 0);
        this->bias_a += state_add.block<3, 1>(12, 0);
        this->gravity += state_add.block<3, 1>(15, 0);
        return *this;
    };

    Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b)
    {
        Matrix<double, DIM_STATE, 1> a;
        M3D rotd(b.rot_end.transpose() * this->rot_end);
        a.block<3, 1>(0, 0) = Log(rotd);
        a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
        a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
        a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
        a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
        a.block<3, 1>(15, 0) = this->gravity - b.gravity;
        return a;
    };

    void resetpose()
    {
        this->rot_end = M3D::Identity();
        this->pos_end = Zero3d;
        this->vel_end = Zero3d;
    }

    M3D rot_end;                              // the estimated attitude (rotation matrix) at the end lidar point
    V3D pos_end;                              // the estimated position at the end lidar point (world frame)
    V3D vel_end;                              // the estimated velocity at the end lidar point (world frame)
    V3D bias_g;                               // gyroscope bias
    V3D bias_a;                               // accelerator bias
    V3D gravity;                              // the estimated gravity acceleration
    Matrix<double, DIM_STATE, DIM_STATE> cov; // states covariance
};

template <typename T>
T rad2deg(T radians)
{
    return radians * 180.0 / PI_M;
}

template <typename T>
T deg2rad(T degrees)
{
    return degrees * PI_M / 180.0;
}

template <typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g,
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)
            rot_kp.rot[i * 3 + j] = R(i, j);
    }
    return move(rot_kp);
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template <typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++)
    {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);

    for (int j = 0; j < point_num; j++)
    {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
        {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

float calc_dist(PointType p1, PointType p2)
{
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        // 如果得到的平面和收敛的特征点的残差大于阈值，该平面失效
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

#endif