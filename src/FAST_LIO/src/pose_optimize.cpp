#include "pose_optimize.h"

void pose_optimize::set_paras(Eigen::Vector3d Point3d, double fx, double fy, double cx, double cy, cv::Mat *img)
{
    m_fx = fx;
    m_fy = fy;
    m_cx = cx;
    m_cy = cy;
    m_img = img;
    m_point = Point3d;
}

void pose_optimize::computeError()
{
    // 只有一个顶点，所以该v就是我们需要的，即_vertices[0]
    const g2o::VertexSE3Expmap *vertex = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
    // 有了顶点就可以知道如何获得该顶点代表的位姿了
    Eigen::Vector3d point_in_cam = vertex->estimate().map(m_point);
    // std::cout<<"opt pose is "<<std::endl;
    // std::cout<<vertex->estimate().to_homogeneous_matrix()<<std::endl;
    // once we have point coordinates in camera frame,we can transform it to pixel frame using formula below
    double u = m_fx * point_in_cam.x() / point_in_cam.z() + m_cx;
    double v = m_fy * point_in_cam.y() / point_in_cam.z() + m_cy;

    if (u - 4 < 0 || (u + 4) > m_img->cols || (v - 4) < 0 || (v + 4) > m_img->rows)
    {
        _error(0, 0) = 0.0;
        // when this edge's level is set to 1,g2o won't compute this edge
        this->setLevel(1);
    }
    else
    {
        _error(0, 0) = get_grayscale_by_pixel(u, v) - _measurement;
    }
}

void pose_optimize::linearizeOplus()
{
    if (level() == 1)
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }

    g2o::VertexSE3Expmap *vertex = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
    // 既然直接用这个顶点的位姿，那么我们传进该顶点的位姿就直接设置成R_1R_2，其中1代表外参，2代表当前雷达帧转到相机帧对应的雷达帧的变换矩阵
    // 所以计算出这个总的变换之后，还需要在外面单独求解某时刻t的雷达坐标变换到另一个时刻相机对应的那个虚拟的雷达坐标
    // 这个变换求出来之后就可以利用t时刻雷达的坐标来计算出对应时刻雷达的坐标，然后比对插值坐标，看看距离咋样，因为我们的原来的对应时刻的雷达坐标就是通过插值算的，只不过现在用优化计算了

    Eigen::Vector3d point_in_cam = vertex->estimate().map(m_point);
    // std::cout<<"opt pose is"<<vertex->estimate().to_homogeneous_matrix()<<std::endl;
    // std::cout<<"point in cam computed by opt pose "<<point_in_cam<<std::endl;
    double x = point_in_cam[0];
    double y = point_in_cam[1];

    double inv_z = 1 / point_in_cam[2];
    double inv_zz = inv_z * inv_z;

    double u = m_fx * x * inv_z + m_cx;
    double v = m_fy * y * inv_z + m_cy;

    // 注意，误差对se3上的扰动的雅可比因为g2o中对se3上李代数的定义是平移在前旋转在后
    Eigen::Matrix<double, 2, 6> jacobian_uv_xi;
    Eigen::Matrix<double, 1, 2> jacobian_pixel;

    jacobian_uv_xi(0, 0) = -m_fx * x * y * inv_zz;
    jacobian_uv_xi(0, 1) = m_fx + m_fx * x * x * inv_zz;
    jacobian_uv_xi(0, 2) = -(m_fx * y * inv_z);
    jacobian_uv_xi(0, 3) = m_fx * inv_z;
    jacobian_uv_xi(0, 4) = 0;
    jacobian_uv_xi(0, 5) = -m_fx * x * inv_zz;

    jacobian_uv_xi(1, 0) = -m_fy - (m_fy * y * y * inv_zz);
    jacobian_uv_xi(1, 1) = m_fy * x * y * inv_zz;
    jacobian_uv_xi(1, 2) = m_fy * x * inv_z;
    jacobian_uv_xi(1, 3) = 0;
    jacobian_uv_xi(1, 4) = m_fy * inv_z;
    jacobian_uv_xi(1, 5) = -m_fy * y * inv_zz;

    // 第一个是对u方向的梯度，第二个是对方向的梯度
    jacobian_pixel(0, 0) = (get_grayscale_by_pixel(u + 1.0, v) - get_grayscale_by_pixel(u - 1.0, v)) / 2.0;
    jacobian_pixel(0, 1) = (get_grayscale_by_pixel(u, v + 1.0) - get_grayscale_by_pixel(u, v - 1.0)) / 2.0;

    _jacobianOplusXi = jacobian_pixel * jacobian_uv_xi;
}

inline float pose_optimize::get_grayscale_by_pixel(float u, float v)
{
    // if(u<=0||v<=0||u>m_img->cols||v>m_img->rows){
    //     std::cout<<"ROWS: "<<m_img->rows<<std::endl;
    //     std::cout<<"COLS: "<<m_img->cols<<std::endl;
    //     std::cout<<"ERRRRRRRRRRRRRRRRRRRRRRROR"<<std::endl;
    //     std::cout<<"UU:"<<u<<std::endl;
    //     std::cout<<"VV:"<<v<<std::endl;
    // }
    // // using bilinear interploation
    // float bbox_min_u = floor(u);
    // float bbox_min_v = floor(v);

    // // m_pixel_buffer.clear();
    // // for(int i=0;i<2;i++){
    // //     for(int j=0;j<2;j++){
    // //         //存储的顺序是 (u,v) (u,v+1) (u+1,v) (u+1,v+1)
    // //         m_pixel_buffer.push_back(Eigen::Vector2f(bbox_min_u+i,bbox_min_v+j));
    // //     }
    // // }

    // // 没考虑左下角的像素在图像左上角、右上角、右下角的情况
    // float xx = u - bbox_min_u;
    // float yy = v - bbox_min_v;
    // float x_gray_1 = xx * (float)m_img->at<uchar>(bbox_min_u, bbox_min_v) + (1 - xx) * (float)m_img->at<uchar>(bbox_min_u, bbox_min_v + 1);
    // float x_gray_2 = xx * (float)m_img->at<uchar>(bbox_min_u, bbox_min_v + 1) + (1 - xx) * (float)m_img->at<uchar>(bbox_min_u + 1, bbox_min_v + 1);

    // float y_gray = yy * x_gray_1 + (1 - yy) * x_gray_2;
    // return y_gray;
    uchar *data = &m_img->data[int(v) * m_img->step + int(u)];
    float xx = u - floor(u);
    float yy = v - floor(v);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[m_img->step] +
        xx * yy * data[m_img->step + 1]);
}