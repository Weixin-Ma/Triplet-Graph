#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

struct p2pFactor 
{
	p2pFactor(Eigen::Vector3d vertex_frame1, Eigen::Vector3d vertex_frame2, double weight)
		: vertex_frame1_(vertex_frame1), vertex_frame2_(vertex_frame2), weight_(weight){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const 
	{

		Eigen::Matrix<T, 3, 1> point1_mat{T(vertex_frame1_.x()), T(vertex_frame1_.y()), T(vertex_frame1_.z())};         
		Eigen::Matrix<T, 3, 1> point2_mat{T(vertex_frame2_.x()), T(vertex_frame2_.y()), T(vertex_frame2_.z())};           

		Eigen::Quaternion<T> q_frame2_to_frame1{q[3], q[0], q[1], q[2]};      //order: wxyz
		Eigen::Matrix<T, 3, 1> t_frame2_to_frame1{t[0], t[1], t[2]};          //order: xyz


        Eigen::Matrix<T, 3, 1> point_frame2_in_frame1;                
        point_frame2_in_frame1 = q_frame2_to_frame1 * point2_mat + t_frame2_to_frame1; 
        
        //differences
        T x_diff =  point_frame2_in_frame1.x() - point1_mat.x();
        T y_diff =  point_frame2_in_frame1.y() - point1_mat.y();
        T z_diff =  point_frame2_in_frame1.z() - point1_mat.z();
        
        residual[0] = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff) * weight_;

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d vertex_frame1, const Eigen::Vector3d vertex_frame2, double weight)
	{
		return (new ceres::AutoDiffCostFunction< 
				p2pFactor, 1, 4, 3>( 
			    new p2pFactor(vertex_frame1, vertex_frame2, weight)));
	}

	Eigen::Vector3d vertex_frame1_, vertex_frame2_;
	double weight_;
};
