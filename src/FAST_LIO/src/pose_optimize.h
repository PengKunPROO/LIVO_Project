#pragma once
#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

using namespace g2o;
using namespace std;

class pose_optimize : public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    pose_optimize(/* args */){};
    //pose_optimize(Eigen::Vector3d Point3d, double fx, double fy, double cx, double cy, cv::Mat *img);
    void set_paras(Eigen::Vector3d Point3d, double fx, double fy, double cx, double cy, cv::Mat& img);
    virtual void computeError();
    virtual void linearizeOplus();
    virtual bool read(istream &in){};
    virtual bool write(ostream &out) const {};

public:
    Eigen::Vector3d m_point;
    double m_fx = 0.0, m_fy = 0.0, m_cx = 0.0, m_cy = 0.0;
    cv::Mat m_img;

protected:
    float get_grayscale_by_pixel(float u, float v);

protected:
    // vector<Eigen::Vector2f> m_pixel_buffer;
    int count = 0;
};
