#pragma once
#include <pthread.h>
#include <chrono>
#include <time.h>
#include <vector>
#include "common_lib.h"
#include <Eigen/StdVector>
#include "pose_optimize.h"

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/slam3d/types_slam3d.h"
#include "var_global.h"
#include "Frame.h"
class FrontEndThread
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    /* data */
    bool isFinished = false;
    // 优化的迭代次数
    int opt_num;
    pthread_t pose_backopt_thread;
    pthread_mutex_t opt_finished;
    pthread_mutex_t pose_modify;
    // 需要优化的位姿
    Frame* m_opt_frame;
    // 某个位姿对应的观测数据，其中位姿buffer和map应是一一对应的
    std::vector<goMeas> m_meas_map;
    static void *multi_thread_ptr(void *arg);
    PointCloudXYZI::Ptr m_local_map;

private:
    Eigen::Matrix3d m_matrixIn_eig;

public:
    FrontEndThread(Frame* opt_frame, int opt_num, std::vector<goMeas> meas_map, const Eigen::Matrix3d& matrixIn_eig);
    ~FrontEndThread();

public:
    void startThread();
    void stopThread();
    bool start_optimize();
    bool is_finished()
    {
        bool finished_flag;
        pthread_mutex_lock(&opt_finished);
        finished_flag = isFinished;
        pthread_mutex_unlock(&opt_finished);
        return finished_flag;
    }
    Eigen::Isometry3d get_opt_pose(){
        Eigen::Isometry3d cache_pose;
        pthread_mutex_lock(&pose_modify);
        cache_pose = m_opt_frame->pose();
        pthread_mutex_unlock(&pose_modify);
        return cache_pose;
    }

    Frame* get_opt_frame(){
        Frame* temp;
        pthread_mutex_lock(&pose_modify);
        temp = m_opt_frame;
        pthread_mutex_unlock(&pose_modify);
        return temp;
    }

    PointCloudXYZI::Ptr get_localMap_ptr(){
        return m_local_map;
    }
};

FrontEndThread::FrontEndThread(Frame* opt_frame, int opt_num, std::vector<goMeas> meas_map, const Eigen::Matrix3d& matrixIn_eig)
{
    m_opt_frame = opt_frame;
    m_meas_map = meas_map;
    m_matrixIn_eig = matrixIn_eig;
    isFinished = false;
    m_local_map = opt_frame->points_map();
    ROS_ERROR("size of m_meas_map:%d---------------------%ld",m_meas_map.size(),(long)pose_backopt_thread);
    startThread();
}

FrontEndThread::~FrontEndThread()
{
    m_local_map->clear();
    m_local_map.reset();
    delete m_opt_frame;
    // 是否需要删除位姿内存指针有待商榷
    if(!is_finished()){
        ROS_ERROR("waiting until current opt thread finished! ---%ld",(long)pose_backopt_thread);
        pthread_join(pose_backopt_thread,NULL);
        stopThread();
    }
    else{
        ROS_ERROR("Thread exit normally! ---%ld",(long)pose_backopt_thread);
        stopThread();
    }
}

void FrontEndThread::startThread()
{
    pthread_mutex_init(&opt_finished, NULL);
    pthread_mutex_init(&pose_modify, NULL);
    pthread_create(&pose_backopt_thread, NULL, multi_thread_ptr, (void *)this);
    printf("Multi thread started ---------------------%ld\n",(long)pose_backopt_thread);
}

void FrontEndThread::stopThread()
{
    pthread_mutex_destroy(&opt_finished);
    pthread_mutex_destroy(&pose_modify);
}

bool FrontEndThread::start_optimize()
{
    Eigen::Isometry3d Tcw = m_opt_frame->pose();
    cv::Mat img;
    cv::cvtColor(m_opt_frame->image(),img,cv::COLOR_BGR2GRAY);
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
    pose->setId(m_opt_frame->ID());
    optimer.addVertex(pose);

    int id = 1;

    
    for (goMeas m : m_meas_map)
    {
        pose_optimize *edge = new pose_optimize();
        edge->set_paras(m.p_lidar, m_matrixIn_eig(0, 0), m_matrixIn_eig(1, 1), m_matrixIn_eig(0, 2), m_matrixIn_eig(1, 2), img);
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        edge->setRobustKernel(rk);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        optimer.addEdge(edge);
    }
    ROS_ERROR("Edges in graph: %lu ---------------------%ld", optimer.edges().size(),(long)pose_backopt_thread);
    optimer.initializeOptimization();
    ROS_ERROR("Edges initialization success!---------------------%ld",(long)pose_backopt_thread);
    optimer.optimize(opt_num);
    Tcw = pose->estimate();
    std::cout << "Tcw aft Optimized---------------------" <<pose_backopt_thread<< std::endl;
    std::cout << Tcw.matrix() << std::endl;
   
    pthread_mutex_lock(&pose_modify);
    m_opt_frame->setPose(Tcw);
    pthread_mutex_unlock(&pose_modify);
    ROS_WARN("FrontEnd Finished Cleanly! Thread-----%ld", (long)pose_backopt_thread);
    pthread_mutex_lock(&opt_finished);
    isFinished = true;
    pthread_mutex_unlock(&opt_finished);
    optimer.clear();
    return true;
}

void *FrontEndThread::multi_thread_ptr(void *arg)
{
    FrontEndThread *handle = (FrontEndThread *)arg;
    handle->start_optimize();
    return nullptr;
}