#pragma once
#include <pthread.h>
#include <chrono>
#include <time.h>
#include <vector>
#include <common_lib.h>
#include "pose_optimize.h"

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/slam3d/types_slam3d.h"

class BackendThread
{
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
    vector<ImgOpt> m_pose_buffer;
    // 某个位姿对应的观测数据，其中位姿buffer和map应是一一对应的
    std::map<int, vector<goMeas>> m_meas_map;
    static void *multi_thread_ptr(void *arg);

private:
    Eigen::Matrix3d m_matrixIn_eig;

public:
    BackendThread(vector<ImgOpt> pose_buffer, int opt_num, std::map<int, vector<goMeas>> meas_map, Eigen::Matrix3d matrixIn_eig);
    ~BackendThread();

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
    vector<Isometry3d> get_opt_pose(){
        int size = m_pose_buffer.size();
        vector<Isometry3d> cache(size);
        pthread_mutex_lock(&pose_modify);
        for(int i=0;i<size;i++){
            cache[i] = m_pose_buffer[i].lidar_img;
        }
        pthread_mutex_unlock(&pose_modify);
    }
};

BackendThread::BackendThread(vector<ImgOpt> pose_buffer, int opt_num, std::map<int, vector<goMeas>> meas_map, Eigen::Matrix3d matrixIn_eig)
{
    m_pose_buffer = pose_buffer;
    m_meas_map = meas_map;
    m_matrixIn_eig = matrixIn_eig;
    isFinished = false;
    startThread();
}

BackendThread::~BackendThread()
{
    // 是否需要删除位姿内存指针有待商榷
    if(!is_finished()){
        ROS_ERROR("waiting until current opt thread finished! ---%ld",(long)pose_backopt_thread);
        pthread_join(pose_backopt_thread,NULL);
        stopThread();
    }
    else{
        stopThread();
    }
}

void BackendThread::startThread()
{
    pthread_mutex_init(&opt_finished, NULL);
    pthread_mutex_init(&pose_modify, NULL);
    pthread_create(&pose_backopt_thread, NULL, multi_thread_ptr, (void *)this);
    printf("Multi thread started \n");
}

void BackendThread::stopThread()
{
    pthread_mutex_destroy(&opt_finished);
    pthread_mutex_destroy(&pose_modify);
}

bool BackendThread::start_optimize()
{
    ROS_WARN("Enter");
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    ROS_WARN("Set Opt");

    ROS_WARN("Reset Buffer");
    int pose_size = m_pose_buffer.size();
    for (int i = 0; i < pose_size; i++)
    {
        // g2o::VertexSE3 *v_p = new g2o::VertexSE3();
        g2o::VertexSE3Expmap *v_p = new g2o::VertexSE3Expmap();
        v_p->setId(i);
        v_p->setEstimate(g2o::SE3Quat(m_pose_buffer[i].lidar_img.rotation(), m_pose_buffer[i].lidar_img.translation()));

        optimizer.addVertex(v_p);
    }

    int id = 1;
    int no_weight_num;
    for (int i = 0; i < opt_num; i++)
    {
        for (goMeas m : m_meas_map.at(i))
        {
            pose_optimize *edge = new pose_optimize();
            edge->set_paras(m.p_lidar, m_matrixIn_eig(0, 0), m_matrixIn_eig(1, 1), m_matrixIn_eig(0, 2), m_matrixIn_eig(1, 2), &m_pose_buffer[i].img_ref);
            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
            rk->setDelta(1.0);
            edge->setRobustKernel(rk);
            edge->setVertex(0, optimizer.vertices()[i]);
            edge->setMeasurement(m.grayscale);
            if (m.weight == 0.0 || m.weight == -1.0)
            {
                edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            }
            else
            {
                edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * (1 / m.weight));
            }

            edge->setId(id++);
            optimizer.addEdge(edge);
        }
    }
    ROS_ERROR("BACKEND-------Edges in graph: %lu !!!!!!!!!!!!!!!!!!!!!!!!!", optimizer.edges().size());

    optimizer.initializeOptimization();
    optimizer.optimize(opt_num); // 可以指定优化步数

    pthread_mutex_lock(&pose_modify);
    for (int i = 0; i < pose_size; i++)
    {
        Isometry3d t_cw = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i))->estimate();
        m_pose_buffer[i].lidar_img = t_cw;
    }
    pthread_mutex_unlock(&pose_modify);
    ROS_WARN("Backend Finished Cleanly! Thread-----%ld", (long)pose_backopt_thread);
    pthread_mutex_lock(&opt_finished);
    isFinished = true;
    pthread_mutex_unlock(&opt_finished);
    pthread_yield();
}

void *BackendThread::multi_thread_ptr(void *arg)
{
    BackendThread *handle = (BackendThread *)arg;
    handle->start_optimize();
    return nullptr;
}