//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
#ifndef TRIPLETGRAPH_H
#define TRIPLETGRAPH_H

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include <numeric>
#include <vector>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <random>

#include <pcl/io/io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h> 

#include "tic_toc.h"
#include "CostFunction.h"

#include "omp.h"

struct semantic_config{        //semantic info for kitti
    std::string obj_class;
    int color_b;
    int color_g;
    int color_r;
};

struct config_para{
    std::string seq_id;
    int file_name_length;
    float voxel_leaf_size;
    int angle_resolution;
    float edge_dis_thr;
    bool rviz_show;                 
    bool label_gt;
    float ransca_threshold;         

    std::string cloud_path;         
    std::string label_path;
    std::string pairs_file;
    std::string pose_gt_file;
    std::string cali_file;       

 
    std::vector<__int16_t> classes_for_graph;
    std::map<__int16_t,float> EuCluster_para;
    std::map<__int16_t,semantic_config> semantic_name_rgb;
    std::map<__int16_t,int> minimun_point_one_instance;
    std::map<__int16_t,float> weights_for_class;
    std::map<__int16_t,double> weights_cere_cost;

    int maxIterations_ransac;
    int cere_opt_iterations_max;
    float percentage_matches_used;
    float similarity_refine_thre;

};

struct my_pair
{
    int label;
    std::string frame1;
    std::string frame2;
};

struct class_combination
{
    __int16_t l_f;
    __int16_t l_m;
    __int16_t l_t;
}; 

struct class_combination_info
{
    std::vector<class_combination> triplet_classes;                                                //{C}_{l^m}
    int dim_combination;                                                                           //N1
    std::map<__int16_t,std::map<__int16_t, std::map<__int16_t,int> > > triplet_to_descriptor_bit;  //C to index of C in {C}_{l^m}
    std::map<int, std::vector<class_combination>> bit_to_triplet;                                  //index in {C}_{l^m} to C
}; 

struct instance_center    //geometric centriod of instance
{
    float x;
    float y;
    float z;
};

struct label_id
{
    __int16_t label;
    __int16_t id;
};

struct instance_result
{
    std::map<__int16_t,std::map<__int16_t, instance_center>>  instance_centriods;
    std::pair< std::map<__int16_t, int>, int> instance_number;   //first: instance_number for each class,  second: total instance number
};

struct Graph_matrixs
{
    std::map<int, label_id> index_2_label_id;                     //index in distance matrix, to label and id of the instance 
    Eigen::MatrixXf dis_matrix;                                  //distance matrix (distance between different vertics)
    Eigen::MatrixXi adj_mat;                                     //adjacency matrix, identify edge
};

struct Descriptors 
{
    std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>>  vertex_descriptors;    //set of vertex descriptors, {Des_{v_j}}
    std::map<__int16_t, Eigen::MatrixXi>  global_descriptor;                         //global descriptor, {Des^{l}}
};

struct match
{
    bool available;          //flage for a match
    __int16_t id;            //id of the matched vertex
    double similarity;       
};


struct match_xyz_label
{
   double x1;
   double y1;
   double z1;
   double x2;
   double y2;
   double z2;
    __int16_t label;
};

struct pose_est
{
    Eigen::Quaterniond ori_Q;
    Eigen::Vector3d trans;

};

struct results_output
{
    int pair_label;

    float similarity;
    float similarity_refined;

    double rte;
    double rre;
    double rre1;

    std::pair< std::map<__int16_t, int>, int> instance_number_1;     //first: instance_number for each class,  second: total instance number
    std::pair< std::map<__int16_t, int>, int> instance_number_2;     //first: instance_number for each class,  second: total instance number
};

struct PointXYZRGBLabelID{                               //custom pointcloud type, including, xyz，rgb，class-label，instanc-ID
	PCL_ADD_POINT4D;	//PointXYZ
	PCL_ADD_RGB;		//PointXYZRGB
    uint16_t label;
    uint16_t id;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW		
}EIGEN_ALIGN16;
//registration custom
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBLabelID,
									(float, x, x)
									(float, y, y)
									(float, z, z)
									(uint8_t, b, b)
									(uint8_t, g, g)
									(uint8_t, r, r)
									(float, rgb, rgb)     //not used
									(uint16_t, label, label)
                                    (uint16_t, id, id)
);

struct pc_results
{
    std::map<__int16_t,std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>>> cluster_pc;
    pcl::PointCloud<pcl::PointXYZRGB> original_pc;
};


class TripletGraph
{     //semantic graph extractor for lidar   
public:
    int frame_count_;

    //Constructor
    TripletGraph(ros::NodeHandle &nh);   

    //run partial frames, without omp, but display
    void run_part();     

    //run total frames, with omp, but not display
    void run_omp();    

    //run single pair
    void run_single_pair();

    //load config
    config_para get_config(std::string config_file_dir);

    //ulti function
    void zfill(std::string& in_str,int len);

    //get pairs
    void get_pairs(std::string pairs_file);

    //get pose ground truth 
    std::vector<std::vector<float>> get_pose_gt(std::string pose_gt_file, std::string cali_file);

    ////for building {C}_{lm}, i.e., first-level bins
    void get_class_combination();

    //calculate similaritry and relative pose
    results_output eval(my_pair);

    //load pointcloud with label and id from KITTI files, frame_index refers to frame-1 or frame-2 in a pair
    pc_results get_pointcloud(std::string frame, int frame_index);

    //get geomertic centriods for instances in frame-1 and frame-2
    instance_result get_instance_center(std::map<__int16_t,std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>>> clustered_pc, int frame_index);
    
    //construct semantic graphs, including distance matrix, adjacency mantrix
    Graph_matrixs build_graph(std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens);

    //get vertex and global descriptors
    Descriptors get_descriptor(Graph_matrixs graph_mats, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens,  int instance_number);
    double get_angle(double x1, double y1, double x2, double y2, double x3, double y3);      //untli function, for calcuating \alpha

    //vertex matching 
    std::map<__int16_t,std::map<__int16_t, match>> get_vertex_matches(Descriptors descriptor1, Descriptors descriptor2);

    //pose estimation, result.first is coarse pose ~T, result.second is optimized pose T*
    std::pair<Eigen::Matrix4d, Eigen::Matrix4d> pose_estimate(std::map<__int16_t,std::map<__int16_t, match>> matches, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens1, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens2);

    //6-DoF pose solver，optimization
    pose_est pose_solver(std::vector<match_xyz_label> matches, Eigen::Quaterniond init_Q, Eigen::Vector3d init_xyz); 

    //6-DoF-pose solver, coarse
    pose_est solver_svd(std::vector<match_xyz_label> matches);     

    //get ground truth relative pose between two frames in a pair
    Eigen::Matrix4d get_relative_pose_gt(my_pair pair);

    //calculate RTE and RRE
    std::vector<double> cal_RPE(Eigen::Matrix4d est_mat, Eigen::Matrix4d gt_mat);

    //similarity between two graphs without projection-based selection
    float cal_similarity(Descriptors descriptor1, Descriptors descriptor2);

    //projection-based selection 
    std::map<__int16_t,std::map<__int16_t, match>> select_matches(std::map<__int16_t,std::map<__int16_t, match>> origianl_matches, Eigen::Matrix4d ested_pose, std::map<__int16_t,std::map<__int16_t, instance_center>>ins_cen1,std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cen2);

    //similarity between two graphs after projection-based selection
    float cal_refined_similarity(Descriptors descriptor1, Descriptors descriptor2, std::map<__int16_t,std::map<__int16_t, match>> filtered_match);

    //showing progress
    void showProgress(float progress);

    //results saving
    void save_results(std::map<int, results_output> results, int pair_number);

    //instance visualization
    visualization_msgs::MarkerArray ins_center_visual(std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens, int frame_index);

    //graph edges visualization
    visualization_msgs::MarkerArray edges_visual(std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens, Graph_matrixs graph_mats, int frame_index);

    //original matches visualization
    visualization_msgs::MarkerArray matches_visual(std::map<__int16_t,std::map<__int16_t, match>> matches,std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens1, std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens2);
    
    //remaining matches visualization
    visualization_msgs::MarkerArray remaining_matches_visual(std::map<__int16_t,std::map<__int16_t, match>> filtered_matches,std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens1, std::map<__int16_t,std::map<__int16_t,instance_center>>ins_cens2);  

    //****************function for sort a vector and return the index************** 
    template <typename T>
    std::vector<int> argsort(const std::vector<T> &v) 
    { 
        std::vector<int> idx(v.size()); 
        iota(idx.begin(), idx.end(), 0); 
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];}); 
        return idx; 
    }

    template <typename D>
    std::vector<int> argsort_s(const std::vector<D> &v) 
    { 
        std::vector<int> idx(v.size());
        iota(idx.begin(), idx.end(), 0); 
        sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];}); 
        return idx; 
    }
    //****************function for sort a vector and return the index************** 

private:
    ros::NodeHandle nh_;                                 
    ros::Publisher pc1_pub_;
    ros::Publisher pc2_pub_;
    ros::Publisher pc2_in_pc1_pub_;                        
    ros::Publisher ins_cen1_pub_;
    ros::Publisher ins_cen2_pub_;
    ros::Publisher edges1_pub_;
    ros::Publisher edges2_pub_;
    ros::Publisher orignial_matches_pub_;
    ros::Publisher remaining_matches_pub_;


    std::vector<my_pair> pairs_;             
};

#endif 
 