//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
#include "./../include/tripletgraph.h"

std::string config_path_;  //path for config file
config_para conf_para_;    //used parameters
std::string results_path_; //folder for results

std::map<__int16_t, class_combination_info> class_combination_infos_;     //all infos for {C}_{lm}, including {C}_{lm}, and related mapping matrices

std::vector<std::vector<float>> gt_pose_;    
std::vector<std::vector<float>> gt_calib_;                              //Pose "Tr" in Kitti，camera to lidar                            
 
float z_offest_ = 25;     //for visualization

std::ofstream f_score_out_;
std::ofstream f_rte_out_;
std::ofstream f_rre_out_;
std::ofstream f_cluster_out_;


TripletGraph::TripletGraph(ros::NodeHandle &nh){
  nh_=nh;
  
  std::cout<<"\033[33;1m*************TripletGraph starts******************\033[0m"<<std::endl;
  std::cout<<std::endl;
  TicToc inital_t;

  nh_.getParam("config_file",config_path_);
  nh_.getParam("results_path",results_path_);

  std::cout<<config_path_<<std::endl;
  conf_para_ = get_config(config_path_);

  get_pairs(conf_para_.pairs_file);

  gt_pose_ = get_pose_gt(conf_para_.pose_gt_file, conf_para_.cali_file);

  get_class_combination();

  //results saving
  f_score_out_  .open(results_path_ +"score.txt");
  f_rte_out_    .open(results_path_ +"rte.txt");
  f_rre_out_    .open(results_path_ +"rre.txt");
  f_cluster_out_.open(results_path_ +"clusters.txt");

  std::cout<<"\033[40;35m[Initalization] consuming time: \033[0m"<<inital_t.toc()<<"ms"<<std::endl;


  pc1_pub_               = nh_.advertise<sensor_msgs::PointCloud2>("pc_1", 1000);
  pc2_pub_               = nh_.advertise<sensor_msgs::PointCloud2>("pc_2", 1000);
  ins_cen1_pub_          = nh_.advertise<visualization_msgs::MarkerArray>("ins_1", 1000); 
  ins_cen2_pub_          = nh_.advertise<visualization_msgs::MarkerArray>("ins_2", 1000);  
  edges1_pub_            = nh_.advertise<visualization_msgs::MarkerArray>("edges_graph1", 10000); 
  edges2_pub_            = nh_.advertise<visualization_msgs::MarkerArray>("edges_graph2", 10000);
  orignial_matches_pub_  = nh_.advertise<visualization_msgs::MarkerArray>("original_matches", 10000);
  remaining_matches_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("remaining_matches", 10000);
  pc2_in_pc1_pub_        = nh_.advertise<sensor_msgs::PointCloud2>("pc2_in_pc1_est", 1000);

  sleep(2);      //for publisher registration
}

void TripletGraph::run_part(){
  int pair_num = 200;
  std::map<int, results_output> results;

  for (int i = 0; i < pair_num; ++i)
  {
    std::cout<<"\033[33;1m**************Current pair: \033[0m"<<pairs_[i].frame1<<" - "<<pairs_[i].frame2<<"\033[33;1m**********************\033[0m"<<std::endl;
    results_output result = eval(pairs_[i]); 
    results[i] = result;

    float progress = (i) * 1.0 / (float)pair_num;
    showProgress(progress);    
  }

  showProgress(1.0);

  save_results(results, pair_num);

  ros::shutdown();
} 

void TripletGraph::run_omp()
{
  conf_para_.rviz_show = false;
  omp_set_num_threads(8);

  int pair_num = pairs_.size();
  //int pair_num = 2000;
  std::map<int, results_output> results;

  #pragma omp parallel for
  for (int i = 0; i < pair_num; ++i)
  {
    std::cout<<"\033[33;1m**************Current pair: \033[0m"<<pairs_[i].frame1<<" - "<<pairs_[i].frame2<<"\033[33;1m**********************\033[0m"<<std::endl;
    results_output result = eval(pairs_[i]); 
    results[i] = result;

    float progress = (i) * 1.0 / (float)pair_num;
    showProgress(progress);    
  }

  showProgress(1.0);

  save_results(results, pair_num);

  ros::shutdown();  
}

void TripletGraph::run_single_pair()
{
  int pair_id =0;
  std::cout<<"\033[33;1m**************Evaluated sequence & pair: \033[0m"<<" seq-"<<conf_para_.seq_id<<" , "<<pairs_[pair_id].frame1<<" - "<<pairs_[pair_id].frame2<<"\033[33;1m**********************\033[0m"<<std::endl;
  conf_para_.rviz_show =true;
  results_output result = eval(pairs_[pair_id]); 
}


config_para TripletGraph::get_config(std::string config_file_dir){
  config_para output;

  auto data_cfg = YAML::LoadFile(config_file_dir);
 
  output.seq_id                  = data_cfg["eval_seq"]["seq_id"].as<std::string>();

  bool label_gt = data_cfg["lable_gt"].as<bool>();          //semantic labels flag, true: ground truht; false: RangeNet++


  if(label_gt)
  {
    output.label_path            = data_cfg["eval_seq"]["dataset_path"].as<std::string>() + output.seq_id  +"/labels/";
  }
  else
  {
    output.label_path            = data_cfg["eval_seq"]["label_prediction_path"].as<std::string>()+ output.seq_id + "/predictions/";
  }


  output.cloud_path              = data_cfg["eval_seq"]["dataset_path"].as<std::string>() + output.seq_id +"/velodyne/";
  output.pose_gt_file            = data_cfg["eval_seq"]["dataset_path"].as<std::string>() + output.seq_id + "/poses.txt";
  output.cali_file               = data_cfg["eval_seq"]["dataset_path"].as<std::string>() + output.seq_id + "/calib.txt";

  output.pairs_file              = data_cfg["eval_seq"]["pairs_file"].as<std::string>() + output.seq_id + ".txt"; 


  output.file_name_length        = data_cfg["file_name_length"].as<int>();
  output.voxel_leaf_size         = data_cfg["voxel_leaf_size"].as<float>();                     //for pointcloud downsample
  output.angle_resolution        = data_cfg["angel_resolution"].as<int>();                      // \theta
  output.edge_dis_thr            = data_cfg["edge_dis_thr"].as<float>();                        // \tau_{edge}
  output.maxIterations_ransac    = data_cfg["max_iterations_ransac"].as<int>();                 
  output.ransca_threshold        = data_cfg["ransac_threshold"].as<float>();    
  output.cere_opt_iterations_max = data_cfg["cere_opt_iterations_max"].as<int>();                           
  output.percentage_matches_used = data_cfg["percentage_matches_used"].as<float>();                       
  output.similarity_refine_thre  = data_cfg["project_select_thresold"].as<float>();          
  output.rviz_show               = data_cfg["rviz_show"].as<bool>();                        
  output.label_gt                = label_gt;

  auto classes_for_graph          =data_cfg["classes_for_graph"];
  auto instance_seg_para          =data_cfg["instance_seg_para"];
  auto class_name                 =data_cfg["labels"];
  auto color_map                  =data_cfg["color_map"];
  auto minimun_point_one_instance =data_cfg["mini_point_one_instance"];
  auto weights_for_class          =data_cfg["weights_for_class"];
  auto weights_for_cere_cost      =data_cfg["weights_for_cere_cost"];


  std::map<__int16_t,std::string> label_name;
  std::vector<__int16_t> class_for_graph;
  std::map<__int16_t,float> EuCluster_para;
  std::map<__int16_t,semantic_config> semantic_name_rgb;
  std::map<__int16_t,int> minimun_point_a_instance;
  std::map<__int16_t,float> classes_weights;
  std::map<__int16_t,double> cere_cost_weights;


  YAML::Node::iterator iter,iter1, iter2, iter4, iter5, iter6;
  iter  =classes_for_graph.begin();
  iter1 =class_name.begin();
  iter2 =instance_seg_para.begin();
  iter4 =minimun_point_one_instance.begin();
  iter5 =weights_for_class.begin();
  iter6 =weights_for_cere_cost.begin();

  //used class, i.e., L
  for (iter;iter!=classes_for_graph.end();iter++) {
    if(iter->second.as<bool>())
    {
      class_for_graph.push_back(iter->first.as<__int16_t>());
    }
  }
  output.classes_for_graph = class_for_graph;

  //label names
  for (iter1;iter1!=class_name.end();iter1++) {
    label_name[iter1->first.as<__int16_t>()] = iter1->second.as<std::string>();
  }  
  
  //paras for instance clustering 
  for (iter2;iter2!=instance_seg_para.end();iter2++) {
    EuCluster_para[iter2->first.as<__int16_t>()] = iter2->second.as<float>();
  }
  output.EuCluster_para = EuCluster_para;

  //paras for instance clustering 
  for (iter4; iter4!=minimun_point_one_instance.end(); iter4++)
  {
    minimun_point_a_instance[iter4->first.as<__int16_t>()] = iter4->second.as<int>();
  }
  output.minimun_point_one_instance = minimun_point_a_instance;

  //weights for different class when calculating similarity between two graphs, we simply let all as 1.0
  float weight_sum = 0.0;
  for (iter5;iter5!=weights_for_class.end();iter5++) {

    if (std::find(class_for_graph.begin(), class_for_graph.end(), iter5->first.as<__int16_t>()) != class_for_graph.end())
    {
      classes_weights[iter5->first.as<__int16_t>()] = iter5->second.as<float>();
      weight_sum = weight_sum + iter5->second.as<float>();
    }
  }
  //normalization
  std::map<__int16_t,float>::iterator iter_weigth;
  iter_weigth = classes_weights.begin();
  for (iter_weigth; iter_weigth!=classes_weights.end(); ++iter_weigth)
  { 
    iter_weigth->second = iter_weigth->second/weight_sum;
  }  
  output.weights_for_class = classes_weights;


  //weights for different class when conducting pose estimation, we simply let all as 1.0
  double weight_sum1 = 0.0;
  for (iter6;iter6!=weights_for_cere_cost.end();iter6++) {

    if (std::find(class_for_graph.begin(), class_for_graph.end(), iter6->first.as<__int16_t>()) != class_for_graph.end())
    {
      cere_cost_weights[iter6->first.as<__int16_t>()] = iter6->second.as<double>();
      weight_sum1 = weight_sum1 + iter6->second.as<double>();
    }
  }

  //normalization
  std::map<__int16_t,double>::iterator iter_weigth1;
  iter_weigth1 = cere_cost_weights.begin();
  for (iter_weigth1; iter_weigth1!=cere_cost_weights.end(); ++iter_weigth1)
  { 
    iter_weigth1->second = iter_weigth1->second/weight_sum1;
  }  
  output.weights_cere_cost = cere_cost_weights;

  //class name in string, and rgb values
  YAML::Node::iterator it;
  for (it = color_map.begin(); it != color_map.end(); ++it)
  {
    semantic_config single_semnatic;
    single_semnatic.obj_class = label_name[it->first.as<__int16_t>()];
    single_semnatic.color_b = it->second[0].as<int>();
    single_semnatic.color_g = it->second[1].as<int>();
    single_semnatic.color_r = it->second[2].as<int>();
    semantic_name_rgb[it->first.as<__int16_t>()] = single_semnatic;
  }
  output.semantic_name_rgb = semantic_name_rgb;

  return output;
}

void TripletGraph::zfill(std::string& in_str,int len)
{
  while (in_str.size() < len)
    {
      in_str = "0" + in_str;
    }
}

void TripletGraph::get_pairs(std::string pairs_file)
{
  std::ifstream f_pairs(pairs_file);

  int num = 1;
  while (1)
  {
    my_pair single_pair;
    std::string sequ1, sequ2;
    int label;
    f_pairs >> sequ1;
    f_pairs >> sequ2;
    f_pairs >> label;
    if (sequ1.empty() || sequ2.empty())
    {
        break;
    }
    zfill(sequ1,conf_para_.file_name_length);
    zfill(sequ2,conf_para_.file_name_length);
    
    single_pair.label  = label;    //flage for negative or positive pair
    single_pair.frame1 = sequ1;
    single_pair.frame2 = sequ2;
    pairs_.push_back(single_pair);

    num++;
  }
  
}

std::vector<std::vector<float>> TripletGraph::get_pose_gt(std::string pose_gt_file, std::string cali_file)
{
  std::vector<std::vector<float>> output;
         
  std::ifstream in(pose_gt_file); 
  std::string line;  
  std::vector<std::vector <std::string>> traj;
  if(in) // if the fiel exist  
  {  
            
    while (getline (in, line)) 
    {   
      std::istringstream ss(line);
      std::string word;
      std::vector<std::string> single_pose;
      while ( ss >> word) 
      {
        single_pose.push_back(word);
      }
      traj.push_back(single_pose);
    }  
  }  
  
  else //if the fiel not exist 
  {  
    std::cout <<"\033[33mCould not load the ground truth traj file!\033[0m"  << std::endl;  
  }

  std::vector<float> pose;
  pose.resize(12);
  for (int i=0; i<traj.size();++i)
  {
    for (size_t j = 0; j < 12; ++j)
      {
        std::stringstream s;
        float f;
        s<<std::fixed<<std::setprecision(9)<<traj[i][j];
        s>>f;
        pose[j] = f;
      }
      output.push_back(pose);
  }
  in.close();


  //load clib file
  std::ifstream in_cali(cali_file); 
  std::string line1;  
  std::vector<std::vector <std::string>> traj1;
  if(in_cali) // if the fiel exist  
  {  
            
    while (getline (in_cali, line1)) 
    {   
      std::istringstream ss1(line1);
      std::string word1;
      std::vector<std::string> single_pose1;
      int bit_count =0;
      while ( ss1 >> word1) 
      {
        if (bit_count>=1)
        {
          single_pose1.push_back(word1);
        }
        bit_count = bit_count + 1;
      }
      traj1.push_back(single_pose1);
    }  
  }  
  else //if the fiel not exist 
  {  
    std::cout <<"\033[33mCould not load the ground truth calibration file!\033[0m"  << std::endl;  
  }

  // std::cout<<"Tr: ";
  std::vector<float> pose1;
  pose1.resize(12);
  for (int i=0; i<traj1.size();++i)
  {
    for (size_t j = 0; j < traj1[i].size(); ++j)
      {
        std::stringstream s1;
        float f1;
        s1<<std::fixed<<std::setprecision(9)<<traj1[i][j];
        s1>>f1;
        pose1[j] = f1;
      }
      gt_calib_.push_back(pose1);
  }
  in_cali.close();

  return output;
}

void TripletGraph::get_class_combination()                  //for building {C}_{lm}, i.e., first-level bins
{
  std::vector<__int16_t> classes_for_graph = conf_para_.classes_for_graph;

  //generate triplet
  if(classes_for_graph.size()>=2)                          //at least two classes are required
  {
    class_combination single_combine;                      //a C in {C}_{lm}

    int combination_amount;                                //combination number for {C}_{lm}, equals to N1
    

    for (int m = 0; m < classes_for_graph.size(); ++m)     //pick l_{m}
    {
      combination_amount=0;
      std::vector<class_combination> triplet_class_combs;  //{C}_{lm}, lm = classes_for_graph[m]
      std::map<__int16_t,std::map<__int16_t, std::map<__int16_t,int> > > triplet_to_descriptor_bit;     //C to index of {C}_{lm} (i.e., first-level bin index)
      std::map<int, std::vector<class_combination>> bit_to_triplet;                                     //index of {C}_{lm}  to C 
      triplet_to_descriptor_bit.clear();
      bit_to_triplet.clear();

      //pick l_{f} and pick l_{m} are the same
      for (int j = 0; j < classes_for_graph.size(); ++j)       
      {
        single_combine.l_m = classes_for_graph[m];         
        single_combine.l_f = classes_for_graph[j];
        single_combine.l_t = classes_for_graph[j];

        triplet_class_combs.push_back(single_combine);                       
        triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t] = combination_amount; 
        bit_to_triplet[combination_amount].push_back(single_combine);  
        //std::cout<<combination_amount<<std::endl;
        combination_amount = combination_amount+1;         
      }
      

      //pick l_{f} and pick l_{m} are different
      for (int f = 0; f < classes_for_graph.size(); ++f)          //pick l_{f}
      {
        std::vector<__int16_t> diff_vertex1 = classes_for_graph;
        for (int l=0; l<f+1; ++l)                                 //delet used l_{f}, for picking l_{t}
        {
          diff_vertex1.erase(diff_vertex1.begin());
        }

        for (int t = 0; t < diff_vertex1.size(); ++t)            //pick l_{t}
        {
          //{l_{f}, l_{m} ,l_{t}}, e.g., {fence, trunk, pole}
          single_combine.l_m = classes_for_graph[m];
          single_combine.l_f = classes_for_graph[f];
          single_combine.l_t = diff_vertex1[t];

          triplet_class_combs.push_back(single_combine);               
          triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t]=combination_amount; 
          bit_to_triplet[combination_amount].push_back(single_combine);     
          
          //{l_{t}, l_{m} ,l_{f}}, e.g., {pole, trunk, fence}
          single_combine.l_m = classes_for_graph[m];
          single_combine.l_t = classes_for_graph[f];
          single_combine.l_f = diff_vertex1[t];
        
          triplet_class_combs.push_back(single_combine);                 
          triplet_to_descriptor_bit[single_combine.l_f][single_combine.l_m][single_combine.l_t]=combination_amount;   
          bit_to_triplet[combination_amount].push_back(single_combine);     


          combination_amount =combination_amount + 1;  //for vertex
        }
      }
      class_combination_infos_[classes_for_graph[m]].triplet_classes=triplet_class_combs;
      class_combination_infos_[classes_for_graph[m]].triplet_to_descriptor_bit=triplet_to_descriptor_bit;
      class_combination_infos_[classes_for_graph[m]].dim_combination=combination_amount;   
      class_combination_infos_[classes_for_graph[m]].bit_to_triplet =bit_to_triplet;
    }

    std::cout<<std::endl;
    
    // // print out first-level bins for different classes
    // std::map<__int16_t, class_combination_info>::iterator iter; 
    // iter=class_combination_infos_.begin();
    // for ( iter; iter != class_combination_infos_.end(); ++iter)
    // {
    //   std::cout<<"Class-"<<iter->first<<" : "<<"combination numbers (i.e., N1)="<<iter->second.dim_combination<<";"<<std::endl;
    //   for (size_t i = 0; i < iter->second.triplet_classes.size(); ++i)
    //   {
    //     std::cout<<"    combin: "<<iter->second.triplet_classes[i].l_f<<"-"<<iter->second.triplet_classes[i].l_m<<"-"<<iter->second.triplet_classes[i].l_t<<
    //     ", correspoidng bit: "<<iter->second.triplet_to_descriptor_bit[iter->second.triplet_classes[i].l_f][iter->second.triplet_classes[i].l_m][iter->second.triplet_classes[i].l_t]<<std::endl;
    //   }
    // }
  }
}

pc_results TripletGraph::get_pointcloud(std::string frame, int frame_index)
{
  //TicToc load_pcs_t;  
  std::string cloud_file, sem_file ;
  cloud_file = conf_para_.cloud_path + frame + ".bin";
  sem_file   = conf_para_.label_path + frame + ".label";

  std::fstream input_point(cloud_file.c_str(), std::ios::in | std::ios::binary);               //pointcloud
  std::fstream input_label(sem_file.c_str(), std::ios::in | std::ios::binary);                 //semantic label

  if(!input_point.good() ||  !input_label.good())
  {
    std::cout << "Could not read file: " << cloud_file <<" or "<< sem_file<<std::endl;
    exit(EXIT_FAILURE);
  }

  input_point.seekg(0, std::ios::beg);                                                              
  input_label.seekg(0, std::ios::beg);
  pcl::PointCloud<PointXYZRGBLabelID>::Ptr frame_points (new pcl::PointCloud<PointXYZRGBLabelID>);
  pcl::PointCloud<PointXYZRGBLabelID>::Ptr points_filtered (new pcl::PointCloud<PointXYZRGBLabelID>);   
  pcl::PointCloud<pcl::PointXYZRGB> original_pc;


  // if (frame_index ==1)       
  // {
    int i;
    for (i=0; input_point.good() && input_label.good() && input_label.peek()!=EOF &&input_point.peek()!=EOF; i++) 
    {
      PointXYZRGBLabelID my_point;                
      pcl::PointXYZRGB show_point;

      float intensity;                                                                     //load xyz
      input_point.read((char *) &my_point.x, 3*sizeof(float));                             //load intensity
      input_point.read((char *) &intensity, sizeof(float));                                //load label
      input_label.read((char *) &my_point.label,sizeof(__int16_t));                        //load id
      input_label.read((char *) &my_point.id,sizeof(__int16_t));

      my_point.r = conf_para_.semantic_name_rgb[my_point.label].color_r;
      my_point.g = conf_para_.semantic_name_rgb[my_point.label].color_g;
      my_point.b = conf_para_.semantic_name_rgb[my_point.label].color_b;

      show_point.x = my_point.x;
      show_point.y = my_point.y;
      show_point.z = my_point.z;
      show_point.r = my_point.r;
      show_point.g = my_point.g;
      show_point.b = my_point.b;


      frame_points->push_back(my_point);
      original_pc.push_back(show_point);
    }


  input_point.close();
  input_label.close();

  //std::cout<<"\033[47;35m      <Load data from bin> consuming time: \033[0m"<<load_pcs_t.toc()<<"ms"<<std::endl;


  //filtering
  //TicToc downsample_pcs_t;  
  float leaf_size = conf_para_.voxel_leaf_size;
  pcl::VoxelGrid<PointXYZRGBLabelID> sor;
  sor.setInputCloud(frame_points);                                          
  sor.setLeafSize(leaf_size, leaf_size, leaf_size); 
  sor.filter(*points_filtered);
  //std::cout<<"\033[47;35m      <Downsample pcs> consuming time: \033[0m"<<downsample_pcs_t.toc()<<"ms"<<std::endl;


  //store point cloud by class
  std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>> pc1;
  for (size_t i = 0; i < points_filtered->points.size(); ++i)
  {
    pc1[points_filtered->points[i].label].points.push_back(points_filtered->points[i]);  
  }


  //********************instance cluster
  //TicToc cluster_instance_t;  
  std::map<__int16_t,std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>>> pc_clustered;
  for (size_t i = 0; i < conf_para_.classes_for_graph.size(); ++i)   //clustering for every used class
  {
    __int16_t class_in_graph = conf_para_.classes_for_graph[i];
    if (pc1.count(class_in_graph) == 1)                            //judge whether current scan existing 3-D points with label class_in_graph
    {
      pcl::search::KdTree<PointXYZRGBLabelID>::Ptr tree (new pcl::search::KdTree<PointXYZRGBLabelID>);   
      tree->setInputCloud(pc1[class_in_graph ].makeShared());                               
      std::vector<pcl::PointIndices> cluster_indices;              
      pcl::EuclideanClusterExtraction<PointXYZRGBLabelID> ec;  

      float cluster_tolerance = conf_para_.EuCluster_para[ class_in_graph ];
      ec.setClusterTolerance (cluster_tolerance);         
      ec.setMinClusterSize (20);
      ec.setMaxClusterSize ( pc1[ class_in_graph ].size() );
      ec.setSearchMethod (tree);
      ec.setInputCloud ( pc1[ class_in_graph ].makeShared());
      ec.extract (cluster_indices);

      //cluster results
      std::vector<pcl::PointIndices>::const_iterator it;
      it = cluster_indices.begin();

      __int16_t new_id=0;
      for (it; it!=cluster_indices.end();++it)
      {
        pcl::PointCloud<PointXYZRGBLabelID> cloud_cluster;
        for (const auto& idx : it->indices)
          cloud_cluster.push_back( pc1[ class_in_graph ][idx] );

        int minimun_points_amount = conf_para_.minimun_point_one_instance[class_in_graph];
        if(cloud_cluster.points.size() >= minimun_points_amount)   
        {
          pc_clustered[class_in_graph][new_id]=cloud_cluster;
          new_id++;
        }  
      }
    }
  }
  //std::cout<<"\033[47;35m      <Cluster pcs> consuming time: \033[0m"<<cluster_instance_t.toc()<<"ms"<<std::endl;

  pc_results output;
  
  output.cluster_pc = pc_clustered;
  output.original_pc = original_pc;

  return output;
}

instance_result TripletGraph::get_instance_center(std::map<__int16_t,std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>>> clustered_pc, int frame_index)
{
  std::map<__int16_t,std::map<__int16_t, instance_center>> instance_cens;

  std::map<__int16_t,std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>>>::iterator iter;
  iter = clustered_pc.begin();

  if (frame_index == 1)
  {
    std::cout<<"Instance in Pc-1: ";
    // std::cout<<"                  ";
  }
  if (frame_index == 2)
  {
    std::cout<<"Instance in Pc-2: ";
    // std::cout<<"                  ";
  }

  std::map<__int16_t, int> ins_amount;
  for (size_t i = 0; i < conf_para_.classes_for_graph.size(); ++i)        //初始化
  {
    ins_amount[conf_para_.classes_for_graph[i]] = 0;
  }

  int instance_amount = 0;
  for (iter; iter != clustered_pc.end(); ++iter)
  {
    std::map<__int16_t, pcl::PointCloud<PointXYZRGBLabelID>>::iterator iter1;
    iter1 = iter->second.begin();

    int amount_single_class = 0;

    std::cout<<conf_para_.semantic_name_rgb[iter->first].obj_class<<" = ";
    instance_center ins_cen;
    for (iter1; iter1!=iter->second.end(); iter1++)
    {
      float x_sum=0.0;
      float y_sum=0.0;
      float z_sum=0.0;
      for (int j=0; j<iter1->second.points.size(); ++j)
      {
        x_sum=x_sum + iter1->second.points[j].x;
        y_sum=y_sum + iter1->second.points[j].y;
        z_sum=z_sum + iter1->second.points[j].z;
      }
      ins_cen.x = x_sum/iter1->second.points.size();
      ins_cen.y = y_sum/iter1->second.points.size();
      ins_cen.z = z_sum/iter1->second.points.size();

        
      instance_cens[iter->first][iter1->first] = ins_cen;

      amount_single_class = amount_single_class + 1;
        
      instance_amount = instance_amount + 1;
    }    
    ins_amount[iter->first] = amount_single_class;
    std::cout<<amount_single_class<<" ; ";
  }

  std::cout<<"total = "<<instance_amount;
  std::cout<<std::endl;
  
  instance_result output;
  std::pair< std::map<__int16_t, int>, int> instance_number;
  instance_number.first  = ins_amount;
  instance_number.second = instance_amount;

  output.instance_centriods = instance_cens;

  output.instance_number = instance_number;

  return output;
}


Graph_matrixs TripletGraph::build_graph(std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens)
{
  Graph_matrixs output;

  std::map<__int16_t,std::map<__int16_t, instance_center>>::iterator iter;
  iter = ins_cens.begin();

  int index=0;
  std::map<int, label_id> index_2_label_id;                       //index of distance matrix to label and id of an instance 
  for (iter; iter!=ins_cens.end(); iter++)
  {
    std::map<__int16_t,instance_center>::iterator iter1;
    iter1=iter->second.begin();

    for (iter1; iter1!=iter->second.end(); ++iter1) 
    {
      index_2_label_id[index].label = iter->first;
      index_2_label_id[index].id    = iter1->first;

      index = index +1;
    }
  }  

  //matrixs valuing
  Eigen::MatrixXi adj_mat;
  adj_mat.resize(index,index);     
  adj_mat.setZero();

  Eigen::MatrixXf dis_matrix;
  dis_matrix.resize(index,index);     
  dis_matrix.setZero();
  for (int row=0; row<index; ++row)
  {
    for (int col = 0; col < index; ++col)
    {
      float d_x = ins_cens[index_2_label_id[row].label][index_2_label_id[row].id].x - ins_cens[index_2_label_id[col].label][index_2_label_id[col].id].x;
      float d_y = ins_cens[index_2_label_id[row].label][index_2_label_id[row].id].y - ins_cens[index_2_label_id[col].label][index_2_label_id[col].id].y;
      float d_z = ins_cens[index_2_label_id[row].label][index_2_label_id[row].id].z - ins_cens[index_2_label_id[col].label][index_2_label_id[col].id].z;
      float dis = sqrt( pow(d_x,2) + pow(d_y,2) + pow(d_z,2) );
      dis_matrix(row,col)= dis;

      if ( dis<=conf_para_.edge_dis_thr)
      {
        adj_mat(row,col) = 1;
      }
      
    }
  }

  output.index_2_label_id = index_2_label_id;
  output.dis_matrix       = dis_matrix;
  output.adj_mat          = adj_mat;
  return output;
}

Descriptors TripletGraph::get_descriptor(Graph_matrixs graph_mats, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens,  int instance_number)
{
  Descriptors output;

  int bin_amount = 180 / conf_para_.angle_resolution;   //i.e., N2

  Eigen::MatrixXf dis_mat = graph_mats.dis_matrix;                               
  std::map<int, label_id> index_2_label_id  = graph_mats.index_2_label_id; 

  int triplet_amount =0;                               //number of constructed triplets, i.e., size of {\Delta}_{v_{j}} 

  std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>> vertex_descriptors;   

  std::map<__int16_t, Eigen::MatrixXi> global_descriptor;     
  std::vector<__int16_t> classes_for_graph = conf_para_.classes_for_graph;
  for (size_t i = 0; i < classes_for_graph.size(); ++i)
  {
    Eigen::MatrixXi single_class_overall_des;          //i.e., Des^{l}
    single_class_overall_des.resize(class_combination_infos_[classes_for_graph[i]].dim_combination, bin_amount);
    single_class_overall_des.setZero();
    global_descriptor[classes_for_graph[i]] = single_class_overall_des;
  }
  
  std::vector<int> instance_index(instance_number);    //index of instances, for extracting triplets
  for (int i = 0; i < instance_number; ++i)
  {
    instance_index[i] = i;
  }

  if(instance_index.size()>=3)
  {
    instance_center first_vertex, mid_vertex, third_vertex;    //i.e., v^f, v^m, v^t
    for(int m=0; m<instance_index.size(); ++m)                 //pick v^m
    {
      __int16_t class_label = index_2_label_id[instance_index[m]].label;     
      __int16_t instance_id = index_2_label_id[instance_index[m]].id;      

      int class_combines_amout=class_combination_infos_[class_label].dim_combination;    //i.e., N1, actually same for all classes

      Eigen::MatrixXi vertex_descriptor;                             //singel vertex descriptor
      vertex_descriptor.setZero(class_combines_amout, bin_amount);   //N1 x N2

      std::vector<int> id_1 = instance_index;
      id_1.erase(id_1.begin()+m);                          //delete picked mid-vertex from id_1
      for (int t=0; t<id_1.size()-1; ++t)                  //pick v^t
      {
        std::vector<int> id_2 = id_1; 
        for (int ll=0; ll<t+1; ++ll)                       //delete picked third-vertex from id_2
        {
          id_2.erase(id_2.begin());
        }
        
        for (int f=0; f<id_2.size(); ++f)                 //pick v^{f}
        {
          if( (graph_mats.adj_mat(id_2[f],instance_index[m])==1) & (graph_mats.adj_mat(id_1[t],instance_index[m])==1) )
          {
            first_vertex= ins_cens[index_2_label_id[id_2[f]].label][index_2_label_id[id_2[f]].id];
            mid_vertex  = ins_cens[index_2_label_id[instance_index[m]].label][index_2_label_id[instance_index[m]].id];
            third_vertex= ins_cens[index_2_label_id[id_1[t]].label][index_2_label_id[id_1[t]].id];

            double angle=get_angle(first_vertex.x, first_vertex.y,third_vertex.x, third_vertex.y, mid_vertex.x, mid_vertex.y);
            int row = class_combination_infos_[class_label].triplet_to_descriptor_bit[index_2_label_id[id_2[f]].label][index_2_label_id[instance_index[m]].label][index_2_label_id[id_1[t]].label];

            int col ;
            if (angle==180)
            {
              col = (int)angle/conf_para_.angle_resolution -1;

            }
            else
            {
              col = (int)angle/conf_para_.angle_resolution;
            }

            // std::cout<<"check point  "<<first_vertex.x<<" " <<first_vertex.y<<" "<<third_vertex.x<<" "<<third_vertex.y<<" "<<mid_vertex.x<<" "<<mid_vertex.y<< " angle:"<<angle<< " ; "<<row <<" "<< col<<std::endl;
            vertex_descriptor(row, col) = vertex_descriptor(row, col)+1;

            triplet_amount = triplet_amount + 1;
            //std::cout<<id_2[f]<<"-"<<instance_index[m]<<"-"<<id_1[t]<<" , angle: "<<angle<< "-degree"<<" ,row:"<<row<<" ;clos:"<<col<<std::endl;    //print {\Delta}_{v_{j}}, it would be better set t_edge smaller or use less class for L to reduce number of triplets for easy checking
          }
        }
      }

      vertex_descriptors[class_label][instance_id] = vertex_descriptor;                       //for vertex descriptor

      global_descriptor[class_label] = global_descriptor[class_label] + vertex_descriptor;    //for overall frame_descriptor
    }
    //std::cout<<"\033[32mThe total amount of cross-instance-triplet: \033[0m"<<cross_triplet_amount_<<std::endl;
  }

  output.vertex_descriptors   = vertex_descriptors; 
  output.global_descriptor    = global_descriptor; 

  return output;
}

double TripletGraph::get_angle(double x1, double y1, double x2, double y2, double x3, double y3)
/*get angle ACB, point C is the center point A(x1,y1) B(x2,y2) C(x3,y3), range: [0, 180]*/
{
  double theta = atan2(x1 - x3, y1 - y3) - atan2(x2 - x3, y2 - y3);
  if (theta >= M_PI)
  theta -= 2 * M_PI;
  if (theta <= -M_PI)
  theta += 2 * M_PI;
  theta = abs(theta * 180.0 / M_PI);
  return theta;
}

std::map<__int16_t,std::map<__int16_t, match>> TripletGraph::get_vertex_matches(Descriptors descriptor1, Descriptors descriptor2)
{
  std::map<__int16_t,std::map<__int16_t, match>> output;

  std::map<__int16_t,std::map<__int16_t, Eigen::MatrixXi>>::iterator iter;
  iter = descriptor1.vertex_descriptors.begin();
  match match_result;   //single match   

  for (iter; iter != descriptor1.vertex_descriptors.end(); ++iter)
  {
    std::map<__int16_t, Eigen::MatrixXi>::iterator iter1;
    iter1 = iter->second.begin();
    
    //check whether instances with class-(iter->first) exist in frame-2
    if (descriptor2.vertex_descriptors.count(iter->first) == 1)      //turnk 2 trunk, pole 2 pole, ....
    {
      std::map<__int16_t, Eigen::MatrixXi> vertex_dess_frame2 = descriptor2.vertex_descriptors[iter->first];   //vertex descriptors in graph-2 with label-(iter->first) 

      for (iter1; iter1 != iter->second.end(); ++iter1)             //vertex in graph-1 (class:iter->first, id:iter1->first)
      {
        Eigen::MatrixXi des_1 = iter1->second;                      //i.e., Des_{v^{1}_{j}}
        
        std::map<__int16_t, Eigen::MatrixXi>::iterator iter2;
        iter2 = vertex_dess_frame2.begin();

        std::vector<double> sims;   
        std::vector<__int16_t> ids; 

        //compare current vertex in graph-1 with all vertices in graph-2 that with same class label
        for (iter2; iter2 != vertex_dess_frame2.end(); ++iter2)
        {
          Eigen::MatrixXi des_2 = iter2->second;            //i.e., Des_{v^{2}_{t}}
          Eigen::MatrixXi dot_multipy;
          
          dot_multipy = des_1.cwiseProduct(des_2);
          int sum_1   = dot_multipy.sum();

          int sum_square1 = des_1.squaredNorm();
          int sum_square2 = des_2.squaredNorm();

          double sim = sum_1/( (double)sqrt(sum_square1) * (double)sqrt(sum_square2)  + 1e-10); 

          sims.push_back(sim);
          ids.push_back(iter2->first);
        }
   
        if (sims.size()>1) 
        {
          std::vector<int> sort_indx = argsort<double>(sims);     //descending sort 
          match_result.id = ids[ sort_indx[0] ];                  //top-1 match
          match_result.available = true;
          match_result.similarity = sims[sort_indx[0]];          
        }
        else if(sims.size()==1)
        {
          match_result.id = ids[0];      
          match_result.available = true;
          match_result.similarity= sims[0];
        }        
        else if(sims.size()<=0)
        {
          match_result.available = false;      
        }   

        output[iter->first][iter1->first] = match_result;
      }
    }

    else
    {
      for (iter1; iter1 != iter->second.end(); ++iter1)
      {
        match_result.available = false;
        output[iter->first][iter1->first] = match_result;
      }
    }
    
  }
  
  return output;
}


pose_est TripletGraph::solver_svd(std::vector<match_xyz_label> matches)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr vertexs_1(new pcl::PointCloud<pcl::PointXYZ>());            //vertices in graph-1
  pcl::PointCloud<pcl::PointXYZ>::Ptr vertexs_2(new pcl::PointCloud<pcl::PointXYZ>());            //vertices in graph-2
  pcl::PointXYZ vertex_in_1, vertex_in_2;

  for (size_t i = 0; i < matches.size(); ++i)
  {
    vertex_in_1.x = matches[i].x1;
    vertex_in_1.y = matches[i].y1;
    vertex_in_1.z = matches[i].z1;
    vertex_in_2.x = matches[i].x2;
    vertex_in_2.y = matches[i].y2;
    vertex_in_2.z = matches[i].z2;
    vertexs_1->points.push_back(vertex_in_1);
    vertexs_2->points.push_back(vertex_in_2);
  }


  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> TESVD;
  pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transformation;
  TESVD.estimateRigidTransformation( *vertexs_2, *vertexs_1, transformation);  

  pose_est output;
  Eigen::Matrix4d relative_T_svd;
  relative_T_svd   << transformation(0,0), transformation(0,1), transformation(0,2), transformation(0,3),
                      transformation(1,0), transformation(1,1), transformation(1,2), transformation(1,3),
                      transformation(2,0), transformation(2,1), transformation(2,2), transformation(2,3),
                      0.0                , 0.0                , 0.0                , 1.0                ;

  Eigen::Quaterniond Q_svd;
  Q_svd = relative_T_svd.block<3,3>(0,0);

  Eigen::Quaterniond rot_q(Q_svd.w(), Q_svd.x(),Q_svd.y(), Q_svd.z());
  Eigen::Vector3d trans(transformation(0,3),transformation(1,3),transformation(2,3));

  output.ori_Q = rot_q;
  output.trans = trans;
  
  return output;
} 


pose_est TripletGraph::pose_solver(std::vector<match_xyz_label> matches, Eigen::Quaterniond init_Q, Eigen::Vector3d init_xyz)
{
  std::map<__int16_t, double> residual_wieght = conf_para_.weights_cere_cost;     

  double para_q[4] = {init_Q.x(), init_Q.y(), init_Q.z(), init_Q.w()};    // set initial value
  double para_t[3] = {init_xyz.x(), init_xyz.y(), init_xyz.z()};          // set initial value
  Eigen::Map<Eigen::Quaterniond> q_2_to_1(para_q);                        
  Eigen::Map<Eigen::Vector3d> t_2_to_1(para_t);                           

  //setting
  ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);                                  
  ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();  //set LocalParameterization，adding operation for Quaternion
  ceres::Problem::Options problem_options;                                                          
  ceres::Problem problem(problem_options);

  //add redisual error
  for (size_t i = 0; i < matches.size(); i++)
  {
    double weight = residual_wieght[matches[i].label];
    Eigen::Vector3d vertex_frame1(matches[i].x1, matches[i].y1, matches[i].z1);
    Eigen::Vector3d vertex_frame2(matches[i].x2, matches[i].y2, matches[i].z2);
    ceres::CostFunction *cost_function = p2pFactor::Create(vertex_frame1, vertex_frame2, weight);   //factory moudel to build cost_function 
    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);                         //add redisual
  }
  

  //when use LocalParameterization，these are required 
  problem.AddParameterBlock(para_q, 4, q_parameterization); // para_q，dim=4
  problem.AddParameterBlock(para_t, 3);                     // para_t，dim=3

  //set colver
  TicToc t_cere_solve;  //solving time 
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;    
  options.max_num_iterations = conf_para_.cere_opt_iterations_max;
  options.minimizer_progress_to_stdout = false; 

  ceres::Solver::Summary summary;

  //start optimization
  ceres::Solve(options, &problem, &summary);

  pose_est pose_result;
  Eigen::Quaterniond rot_q(q_2_to_1.w(), q_2_to_1.x(),q_2_to_1.y(), q_2_to_1.z());
  Eigen::Vector3d trans(t_2_to_1.x(),t_2_to_1.y(),t_2_to_1.z());

  pose_result.ori_Q = rot_q;
  pose_result.trans = trans;
  
  return pose_result;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> TripletGraph::pose_estimate(std::map<__int16_t,std::map<__int16_t, match>> matches, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens1, std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens2)
{
  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = matches.begin();                                 

  //load matches
  std::vector<int> match_ids;                     
  std::vector<match_xyz_label> matched_pairs;
  int match_xyz_count = 0;
  for (iter; iter != matches.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 = iter->second.begin();
    //std::cout<<"class: "<<iter11->first<<std::endl;

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (iter1->second.available)
      {
        match_xyz_label one_matche;
        one_matche.label = iter->first;

        //vertex's coordinate in lidar frame1
        one_matche.x1 =ins_cens1[iter->first][iter1->first].x;
        one_matche.y1 =ins_cens1[iter->first][iter1->first].y;
        one_matche.z1 =ins_cens1[iter->first][iter1->first].z;

        //matched vertex's coordinate in lidar frame2
        one_matche.x2 =  ins_cens2[iter->first][iter1->second.id].x;
        one_matche.y2 =  ins_cens2[iter->first][iter1->second.id].y;
        one_matche.z2 =  ins_cens2[iter->first][iter1->second.id].z;

        match_ids.push_back(match_xyz_count);
        matched_pairs.push_back(one_matche);
        match_xyz_count = match_xyz_count + 1;
      }
    }
  //std::cout<<std::endl;  
  }

  // Create a random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, match_xyz_count - 1);

  //number of matches needed for RANSAC solving
  int sample_amount = int (conf_para_.percentage_matches_used * match_xyz_count);
  if (sample_amount < 3)
  {
    std::cout<<"Error, more matches are needed!"<<std::endl;       
  }

  int best_count = 0;
  Eigen::Matrix4d best_pose;  
  best_pose << 1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0;

  std::vector<match_xyz_label> best_inliners;

  //std::cout<<"total match amout: "<<match_xyz_count<<std::endl;

  //RANSAC + SVD 
  TicToc t_ransac_svd;
  for (int i = 0; i < conf_para_.maxIterations_ransac; ++i) 
  {
    std::vector<match_xyz_label> matche_samples;
    std::set<int> unique_nums;                // A set to store unique random numbers

    while(unique_nums.size()<sample_amount)
    {
      int num = distrib(gen);
      if(unique_nums.find(num) == unique_nums.end())
      {
        unique_nums.insert(num);
      }
    }

    //sample matches for current iteration 
    for(auto num : unique_nums)
    {
      matche_samples.push_back(matched_pairs[num]);
    }

    //SVD solving
    pose_est pose_2_to_1 = solver_svd(matche_samples);
    Eigen::Matrix3d rot_mat = pose_2_to_1.ori_Q.matrix();
    Eigen::Matrix4d pose_mat; 
    pose_mat << rot_mat(0,0), rot_mat(0,1), rot_mat(0,2), pose_2_to_1.trans.x(),
                rot_mat(1,0), rot_mat(1,1), rot_mat(1,2), pose_2_to_1.trans.y(),
                rot_mat(2,0), rot_mat(2,1), rot_mat(2,2), pose_2_to_1.trans.z(),
                0.0         , 0.0         , 0.0         , 1.0                  ;
    
    
    //mark inliers and outliers
    int count = 0; 
    std::vector<match_xyz_label> current_inliners;
    for (size_t i = 0; i < matched_pairs.size(); ++i)
    {
      Eigen::Vector4d vertex_frame1(matched_pairs[i].x1, matched_pairs[i].y1, matched_pairs[i].z1, 1.0);
      Eigen::Vector4d vertex_frame2_t(matched_pairs[i].x2, matched_pairs[i].y2, matched_pairs[i].z2, 1.0);
      vertex_frame2_t = pose_mat * vertex_frame2_t; 

      double project_dis = sqrt(pow(vertex_frame1[0]-vertex_frame2_t[0], 2)+pow(vertex_frame1[1]-vertex_frame2_t[1], 2)+pow(vertex_frame1[2]-vertex_frame2_t[2], 2) );
      if (project_dis < conf_para_.ransca_threshold)
      {
        count++;
        current_inliners.push_back(matched_pairs[i]);
      } 
    }

    //update
    if (count > best_count)
    {
      best_pose = pose_mat;     //coarse pose
      best_count= count;
      best_inliners.swap(current_inliners);
    }
    
  }

  //************* Pose optimization
  Eigen::Matrix3d initial_rot_mat  = best_pose.block<3,3>(0,0);
  Eigen::Vector3d initial_eulerAngle = initial_rot_mat.eulerAngles(0,1,2); //Z-Y-X, RPY
  Eigen::Quaterniond initial_rot_Q = Eigen::Quaterniond(initial_rot_mat);
  Eigen::Vector3d initial_xyz (best_pose(0,3), best_pose(1,3), best_pose(2,3));

  // std::cout <<"         Ransac: translation(XYZ): " << best_pose(0, 3)<<" " << best_pose(1, 3)<<" " <<best_pose(2, 3)<<" " <<std::endl; 
  // std::cout <<"         Ransac: rotation(RPY):    " << initial_eulerAngle[0]/M_PI *180<<" " << initial_eulerAngle[1]/M_PI *180<<" " <<initial_eulerAngle[2]/M_PI *180<<std::endl;
  // std::cout<<"\033[34;1m         Ransac: Runtime:\033[0m "<<t_ransac_svd.toc()<<"ms"<<std::endl;

  //with best inliners，use ceres to further optimize
  TicToc t_ceres;
  pose_est pose_optimized = pose_solver(best_inliners, initial_rot_Q, initial_xyz);

  Eigen::Matrix4d optimized_pose; 
  Eigen::Matrix3d optimized_rot = pose_optimized.ori_Q.matrix();
  Eigen::Vector3d optimized_eulerAngle = optimized_rot.eulerAngles(0,1,2); //Z-Y-X, RPY
  optimized_pose << optimized_rot(0,0), optimized_rot(0,1), optimized_rot(0,2), pose_optimized.trans.x(),
                    optimized_rot(1,0), optimized_rot(1,1), optimized_rot(1,2), pose_optimized.trans.y(),
                    optimized_rot(2,0), optimized_rot(2,1), optimized_rot(2,2), pose_optimized.trans.z(),
                    0.0               , 0.0               , 0.0               , 1.0                     ;

  std::pair<Eigen::Matrix4d, Eigen::Matrix4d> output;
  output.first  = best_pose;
  output.second = optimized_pose;

  std::cout <<"Pose estimation: translation(XYZ): " << optimized_pose(0, 3)<<" " << optimized_pose(1, 3)<<" " <<optimized_pose(2, 3)<<" " <<std::endl; 
  std::cout <<"Pose estimation: rotation(RPY):    " << optimized_eulerAngle[0]/M_PI *180<<" " << optimized_eulerAngle[1]/M_PI *180<<" " <<optimized_eulerAngle[2]/M_PI *180<<std::endl;
  //std::cout<<"\033[34;1m Ceres_opt: Runtime:\033[0m "<<t_ceres.toc()<<"ms"<<std::endl;

  return output;   
}


Eigen::Matrix4d TripletGraph::get_relative_pose_gt(my_pair pair)
{
  std::string str1 = pair.frame1;
  std::string str2 = pair.frame2;

  int frame1_index = 0;
  int frame2_index = 0;

  std::istringstream ss1(str1);
  ss1 >> frame1_index;
  std::istringstream ss2(str2);
  ss2 >> frame2_index;

  // std::cout<<"Frame1 pose: "<<frame1_index<<std::endl;
  // for (size_t i = 0; i < 12; ++i)
  // {
  //   std::cout<<std::fixed<<std::setprecision(9)<<gt_pose_[frame1_index][i]<<" ";
  // }
  // std::cout<<std::endl;
  // std::cout<<"Frame2 pose: "<<frame2_index<<std::endl;
  // for (size_t j = 0; j < 12; ++j)
  // {
  //   std::cout<<std::fixed<<std::setprecision(9)<<gt_pose_[frame2_index][j]<<" ";
  // }
  // std::cout<<std::endl;  

  Eigen::Matrix4d pose_frame1, pose_frame2, relative_pose, Tr;
  pose_frame1 <<  gt_pose_[frame1_index][0], gt_pose_[frame1_index][1], gt_pose_[frame1_index][2] , gt_pose_[frame1_index][3] ,
                  gt_pose_[frame1_index][4], gt_pose_[frame1_index][5], gt_pose_[frame1_index][6] , gt_pose_[frame1_index][7] ,
                  gt_pose_[frame1_index][8], gt_pose_[frame1_index][9], gt_pose_[frame1_index][10], gt_pose_[frame1_index][11],               
                  0.0                      , 0.0                      , 0.0                       , 1.0                       ;

  pose_frame2 <<  gt_pose_[frame2_index][0], gt_pose_[frame2_index][1], gt_pose_[frame2_index][2] , gt_pose_[frame2_index][3] ,
                  gt_pose_[frame2_index][4], gt_pose_[frame2_index][5], gt_pose_[frame2_index][6] , gt_pose_[frame2_index][7] ,
                  gt_pose_[frame2_index][8], gt_pose_[frame2_index][9], gt_pose_[frame2_index][10], gt_pose_[frame2_index][11],               
                  0.0                      , 0.0                      , 0.0                       , 1.0                       ;

  Tr  <<gt_calib_[4][0], gt_calib_[4][1], gt_calib_[4][2] , gt_calib_[4][3] , 
        gt_calib_[4][4], gt_calib_[4][5], gt_calib_[4][6] , gt_calib_[4][7] , 
        gt_calib_[4][8], gt_calib_[4][9], gt_calib_[4][10], gt_calib_[4][11], 
        0.0            , 0.0            , 0.0             , 1.0             ;    

  relative_pose = Tr.inverse() * pose_frame1.inverse() * pose_frame2 * Tr;

  Eigen::Matrix3d rot_gt = relative_pose.block<3,3>(0,0);
  Eigen::Vector3d gt_eulerAngle = rot_gt.eulerAngles(0,1,2); //Z-Y-X, RPY

  // std::cout<<"\033[34;1mPose ground truth: \033[0m"<<std::endl;
  // std::cout <<"                Translation(XYZ): " << relative_pose(0,3)<<" " << relative_pose(1,3)<<" " <<relative_pose(2,3)<<" " <<std::endl; 
  // std::cout <<"                Rotation(RPY): " << gt_eulerAngle[0]/M_PI *180<<" " << gt_eulerAngle[1]/M_PI *180<<" " <<gt_eulerAngle[2]/M_PI *180<<std::endl;

  return relative_pose;
}



std::vector<double> TripletGraph::cal_RPE(Eigen::Matrix4d est_mat, Eigen::Matrix4d gt_mat)
{
  std::vector<double> output;
  double RTE = 0.0;
  double RRE = 0.0;
  double RRE1 = 0.0;  

  RTE = sqrt( (est_mat(0,3)-gt_mat(0,3))*(est_mat(0,3)-gt_mat(0,3)) + (est_mat(1,3)-gt_mat(1,3))*(est_mat(1,3)-gt_mat(1,3)) + (est_mat(2,3)-gt_mat(2,3))*(est_mat(2,3)-gt_mat(2,3))  );

  Eigen::Matrix3d est_R = est_mat.block<3,3>(0,0);
  Eigen::Matrix3d gt_R  = gt_mat.block<3,3>(0,0);

  double a = ((est_R.transpose() * gt_R).trace() - 1) * 0.5;
  double aa= std::max(std::min(a,1.0), -1.0);
  RRE = acos(aa)*180/M_PI;

  double b = ((gt_R.inverse() * est_R).trace() - 1) * 0.5;
  double bb= std::max(std::min(b,1.0), -1.0);
  RRE1 = acos(bb)*180/M_PI;

  output.push_back(RTE);
  output.push_back(RRE);
  output.push_back(RRE1);

  return output;
}

std::map<__int16_t,std::map<__int16_t, match>> TripletGraph::select_matches(std::map<__int16_t,std::map<__int16_t, match>> origianl_matches, Eigen::Matrix4d ested_pose, std::map<__int16_t,std::map<__int16_t, instance_center>>ins_cen1,std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cen2)
{
  std::map<__int16_t,std::map<__int16_t, match>> refined_match;

  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = origianl_matches.begin();
  int original_matches_amount = 0;
  int filtered_matches_amount = 0;
  for (iter; iter != origianl_matches.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 = iter->second.begin();
    match refine_match;
    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      refine_match = iter1->second;
      if (iter1->second.available)
      {
        original_matches_amount = original_matches_amount +1;
        instance_center ins_in_1 = ins_cen1[iter->first][iter1->first];
        Eigen::Vector4d vertex1(ins_in_1.x, ins_in_1.y, ins_in_1.z ,1.0);

        instance_center ins_in_2 = ins_cen2[iter->first][iter1->second.id];
        Eigen::Vector4d vertex2(ins_in_2.x, ins_in_2.y, ins_in_2.z ,1.0);

        Eigen::Vector4d ver2_in_frame1 = ested_pose * vertex2;

        double diff =sqrt( (vertex1.x()-ver2_in_frame1.x())*(vertex1.x()-ver2_in_frame1.x()) + (vertex1.y()-ver2_in_frame1.y())*(vertex1.y()-ver2_in_frame1.y()) + (vertex1.z()-ver2_in_frame1.z())*(vertex1.z()-ver2_in_frame1.z())  );

        if (diff>= conf_para_.similarity_refine_thre)
        {
          refine_match.available = false;
          filtered_matches_amount = filtered_matches_amount +1;
        }
      }

      refined_match[iter->first][iter1->first] = refine_match;
    }
  }
  std::cout<<"original matches amount = "<<original_matches_amount<<"; matches amount for global descriptor="<< original_matches_amount-filtered_matches_amount<<std::endl;
  return refined_match;
}


float TripletGraph::cal_similarity(Descriptors descriptor1, Descriptors descriptor2)
{
  std::map<__int16_t, float> same_class_similarities;
  std::map<__int16_t, Eigen::MatrixXi>::iterator  iter;
  iter = descriptor1.global_descriptor.begin();

  for (iter; iter != descriptor1.global_descriptor.end(); ++iter)
  {
    Eigen::MatrixXi descri1 = iter->second;
    Eigen::MatrixXi descri2 = descriptor2.global_descriptor[iter->first];

    Eigen::MatrixXi dot_multipy;
          
    dot_multipy = descri1.cwiseProduct(descri2); 
    int sum_1   = dot_multipy.sum();

    int sum_square1 = descri1.squaredNorm();
    int sum_square2 = descri2.squaredNorm();

    float sim = sum_1/( (float)sqrt(sum_square1) * (float)sqrt(sum_square2) + 1e-10);

    same_class_similarities[iter->first] = sim;
  }
  
  //fianl similarity
  std::map<__int16_t, float>::iterator sim_iter;
  sim_iter = same_class_similarities.begin();

  float sim = 0.0;
  for (sim_iter; sim_iter != same_class_similarities.end(); ++sim_iter)
  {
    sim = sim + conf_para_.weights_for_class[sim_iter->first] * sim_iter->second;
  }

  return sim;
}


float TripletGraph::cal_refined_similarity(Descriptors descriptor1, Descriptors descriptor2, std::map<__int16_t,std::map<__int16_t, match>> filtered_match)
{
  int bin_amount = 180 / conf_para_.angle_resolution;

  std::map<__int16_t, int> qualified_counts;                         

  std::map<__int16_t, Eigen::MatrixXi>  overall_descriptors_1;    
  std::map<__int16_t, Eigen::MatrixXi>  overall_descriptors_2;    
  for (size_t i = 0; i < conf_para_.classes_for_graph.size(); ++i)  
  {
    qualified_counts[conf_para_.classes_for_graph[i]] = 0;

    Eigen::MatrixXi single_class_overall_des;
    single_class_overall_des.resize(class_combination_infos_[conf_para_.classes_for_graph[i]].dim_combination, bin_amount);
    single_class_overall_des.setZero();
    overall_descriptors_1[conf_para_.classes_for_graph[i]] = single_class_overall_des;
    overall_descriptors_2[conf_para_.classes_for_graph[i]] = single_class_overall_des;
  }

  //claculate {Des^l} for matches after projection selection operation
  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = filtered_match.begin();
  int qualified_count_total = 0;

  for (iter; iter != filtered_match.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 = iter->second.begin();
    int qualified_count_one_class =0;

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (iter1->second.available)
      {   
        overall_descriptors_1[iter->first] = overall_descriptors_1[iter->first] + descriptor1.vertex_descriptors[iter->first][iter1->first];
        overall_descriptors_2[iter->first] = overall_descriptors_2[iter->first] + descriptor2.vertex_descriptors[iter->first][iter1->second.id];

        qualified_count_one_class = qualified_count_one_class + 1;
        qualified_count_total     = qualified_count_total + 1;
      } 
    }

    qualified_counts[iter->first] = qualified_count_one_class;
  }

  //for each class, calculate overall similarity 
  std::map<__int16_t, float> overall_similarities;
  std::map<__int16_t, Eigen::MatrixXi>::iterator  iter_2;
  iter_2 = overall_descriptors_1.begin();

  for (iter_2; iter_2 != overall_descriptors_1.end(); ++iter_2)
  {
    Eigen::MatrixXi des_1 = iter_2->second;
    Eigen::MatrixXi des_2 = overall_descriptors_2[iter_2->first];

    Eigen::MatrixXi dot_multipy;
          
    dot_multipy = des_1.cwiseProduct(des_2);  
    int sum_1   = dot_multipy.sum();

    int sum_square1 = des_1.squaredNorm();
    int sum_square2 = des_2.squaredNorm();

    float sim = sum_1/( (float)sqrt(sum_square1) * (float)sqrt(sum_square2) + 1e-10);

    overall_similarities[iter_2->first] = sim;

    //std::cout<<"class-"<<iter_2->first<<" , sim="<<sim <<" , quali count="<<qualified_counts[iter_2->first]<<std::endl;
  }

  //cal final similarity
  std::map<__int16_t, float>::iterator sim_iter;
  sim_iter = overall_similarities.begin();

  float sim = 0.0;
  for (sim_iter; sim_iter != overall_similarities.end(); ++sim_iter)
  {
    float penalty_factor = 1.0;

    if (qualified_counts[sim_iter->first] == 0)         
    {
      penalty_factor = 0.0;
    }
    //std::cout<<"class-"<<sim_iter->first<<" , penalty="<<penalty_factor<<std::endl;
    sim = sim + penalty_factor * conf_para_.weights_for_class[sim_iter->first] * sim_iter->second;
  }

  return sim;
}

void TripletGraph::save_results(std::map<int, results_output> results, int pair_number)
{
  std::cout<<"saving results...."<<std::endl;

  //first line, statement for instance clusters
  std::map<__int16_t, int>::iterator iter_0;
  iter_0 = results[0].instance_number_1.first.begin();
  f_cluster_out_<<"[PC-1: ";
  for (iter_0; iter_0 != results[0].instance_number_1.first.end(); ++iter_0)
  {
    f_cluster_out_<<conf_para_.semantic_name_rgb[iter_0->first].obj_class<<" ";
  }
  iter_0 = results[0].instance_number_1.first.begin();
  f_cluster_out_<<"total] [PC-2: ";
  for (iter_0; iter_0 != results[0].instance_number_1.first.end(); ++iter_0)
  {
    f_cluster_out_<<conf_para_.semantic_name_rgb[iter_0->first].obj_class<<" ";
  }
  f_cluster_out_<<"total]"<<std::endl;

  for (int i = 0; i < pair_number; ++i)
  {
    //save_similarity
    f_score_out_ <<std::fixed<<std::setprecision(9)<< results[i].similarity << " " <<results[i].similarity_refined<<" " << results[i].pair_label << std::endl;

    //save rte and rre
    f_rte_out_<<std::fixed<<std::setprecision(9)<< results[i].rte<< std::endl;
    f_rre_out_<<std::fixed<<std::setprecision(9)<< results[i].rre<<" "<<results[i].rre1 <<std::endl;

    //***save instance result
    std::map<__int16_t, int> instance_number_1 = results[i].instance_number_1.first;
    std::map<__int16_t, int> instance_number_2 = results[i].instance_number_2.first;

    //instance cluster result in pc1
    std::map<__int16_t, int>::iterator iter;
    iter = instance_number_1.begin();
    for (iter; iter != instance_number_1.end(); ++iter)
    {
      f_cluster_out_<<iter->second<<" ";
    }
    f_cluster_out_<<results[i].instance_number_1.second<<" ";

    //instance cluster result in pc2
    std::map<__int16_t, int>::iterator iter2;
    iter2 = instance_number_2.begin();
    for (iter2; iter2 != instance_number_2.end(); ++iter2)
    {
      f_cluster_out_<<iter2->second<<" ";
    }
    f_cluster_out_<< results[i].instance_number_2.second<<std::endl;
       
  }

  f_score_out_.close();
  f_rte_out_.close();
  f_rre_out_.close();
  f_cluster_out_.close();
  std::cout<<"results saved!!"<<std::endl;
}


//**************visualization function part
visualization_msgs::MarkerArray TripletGraph::ins_center_visual(std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens, int frame_index)
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker marker;

  //set marker orientation
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  //set marker scale
  marker.scale.x = 2.0; 
  marker.scale.y = 2.0;
  marker.scale.z = 2.0;

  //decide the color of the marker
  marker.color.a = 1.0; // Don't forget to set the alpha!

  float z_offest;

  std::string name_space;

  if (frame_index == 1)
  {
    name_space  = "ins_1";
    z_offest = 0.0;
  }
  if (frame_index == 2)
  {
    name_space  = "ins_2";
    z_offest = z_offest_;
  }  

  //************** add markers
  std::map<__int16_t,std::map<__int16_t,instance_center>>::iterator iter;
  iter = ins_cens.begin();

  int id_index    = 0;
  for (iter; iter != ins_cens.end(); iter++)
  {
    std::map<__int16_t,instance_center>::iterator iter1;
    iter1 = iter->second.begin();

    for (iter1; iter1 != iter->second.end(); iter1++)
    {
      //basic para
      marker.header.frame_id = "tripletgraph";
      marker.header.stamp    =ros::Time::now();
      marker.ns = name_space;
      marker.type = visualization_msgs::Marker::SPHERE;

      //set marker action
      marker.action   = visualization_msgs::Marker::ADD;
      marker.lifetime = ros::Duration();//(sec,nsec),0 forever

      //~~~~~~~~~~~~~~~~instance center
      marker.id=id_index;                       
      //set marker position
      marker.pose.position.x = iter1->second.x;
      marker.pose.position.y = iter1->second.y;
      marker.pose.position.z = iter1->second.z + z_offest;

      marker.color.r = (float)conf_para_.semantic_name_rgb[iter->first].color_r/255;
      marker.color.g = (float)conf_para_.semantic_name_rgb[iter->first].color_g/255;
      marker.color.b = (float)conf_para_.semantic_name_rgb[iter->first].color_b/255;

      output.markers.push_back(marker);

      id_index = id_index + 1;
    }
  }

  return output;
}

visualization_msgs::MarkerArray TripletGraph::edges_visual(std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens, Graph_matrixs graph_mats, int frame_index)
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker edge;
              
  edge.color.a=0.5;
  edge.color.r=1.0;         
  edge.color.g=0.0;         
  edge.color.b=0.0;         

  edge.pose.orientation.w=1.0;
  edge.scale.x=0.1;               

  Eigen::MatrixXi adj_mat = graph_mats.adj_mat;          

  float offest_z;            
  std::string name_space;
  if (frame_index == 1)
  {
    offest_z =0;  
    name_space = "edges_graph1";
  }
  else if(frame_index == 2)
  {
    offest_z = z_offest_;
    name_space = "edges_graph2";
  }    


  int id_indx =0;
  for (int i=0; i<adj_mat.rows(); ++i)
  {
    for (int j=i; j<adj_mat.cols(); ++j)
    {
      if(adj_mat(i,j)==1)
      {
        edge.header.frame_id = "tripletgraph";
        edge.header.stamp=ros::Time::now();
        edge.ns = name_space;
        edge.type = visualization_msgs::Marker::LINE_LIST;     

        //set marker action
        edge.action = visualization_msgs::Marker::ADD;
        edge.lifetime = ros::Duration();//(sec,nsec),0 forever

        edge.id= id_indx;                                       
        geometry_msgs::Point p1, p2;
        p1.x = ins_cens[graph_mats.index_2_label_id[i].label][graph_mats.index_2_label_id[i].id].x;
        p1.y = ins_cens[graph_mats.index_2_label_id[i].label][graph_mats.index_2_label_id[i].id].y;
        p1.z = ins_cens[graph_mats.index_2_label_id[i].label][graph_mats.index_2_label_id[i].id].z+offest_z;

        p2.x = ins_cens[graph_mats.index_2_label_id[j].label][graph_mats.index_2_label_id[j].id].x;
        p2.y = ins_cens[graph_mats.index_2_label_id[j].label][graph_mats.index_2_label_id[j].id].y;
        p2.z = ins_cens[graph_mats.index_2_label_id[j].label][graph_mats.index_2_label_id[j].id].z+offest_z;

        edge.points.push_back(p1);
        edge.points.push_back(p2);

        output.markers.push_back(edge);

        id_indx = id_indx + 1;
      }
    }
  }

  return output;
}

visualization_msgs::MarkerArray TripletGraph::matches_visual(std::map<__int16_t,std::map<__int16_t, match>> matches, std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens1,  std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens2)
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker single_match;

  single_match.color.a = 0.6;
  single_match.color.r = 100/255.0;    
  single_match.color.g = 149/255.0;         
  single_match.color.b = 237/255.0;    

  single_match.pose.orientation.w=1.0;
  single_match.scale.x=0.1;             

  single_match.header.frame_id = "tripletgraph";
  single_match.ns = "original_matches";
  single_match.type = visualization_msgs::Marker::LINE_LIST;   

  //set marker action
  single_match.action = visualization_msgs::Marker::ADD;
  single_match.lifetime = ros::Duration();//(sec,nsec),0 forever


  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter;
  iter = matches.begin();

  int matches_index = 0;
  for (iter; iter != matches.end(); ++iter)
  {
    std::map<__int16_t, match>::iterator iter1;
    iter1 =  iter->second.begin();

    for (iter1; iter1 != iter->second.end(); ++iter1)
    {
      if (iter1->second.available)     
      {
        single_match.header.stamp=ros::Time::now();

        single_match.id= matches_index;          

        geometry_msgs::Point p1, p2;
        p1.x = ins_cens1[iter->first][iter1->first].x;
        p1.y = ins_cens1[iter->first][iter1->first].y;
        p1.z = ins_cens1[iter->first][iter1->first].z;

        p2.x = ins_cens2[iter->first][iter1->second.id].x;
        p2.y = ins_cens2[iter->first][iter1->second.id].y;
        p2.z = ins_cens2[iter->first][iter1->second.id].z + z_offest_;

        single_match.points.push_back(p1);
        single_match.points.push_back(p2);

        output.markers.push_back(single_match);        

        matches_index = matches_index + 1;
      }
    }
  }

  return output;
}

visualization_msgs::MarkerArray TripletGraph::remaining_matches_visual (std::map<__int16_t,std::map<__int16_t, match>> filtered_matches,std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens1, std::map<__int16_t,std::map<__int16_t,instance_center>>  ins_cens2)  
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker single_match;

  single_match.color.a = 1.0;
  single_match.color.r = 0.0; 
  single_match.color.g = 1.0;          
  single_match.color.b = 0.0;     

  single_match.pose.orientation.w=1.0;
  single_match.scale.x=0.15;                    

  single_match.header.frame_id = "tripletgraph";
  single_match.ns = "remaining_matches";
  single_match.type = visualization_msgs::Marker::LINE_LIST;     

  //set marker action
  single_match.action = visualization_msgs::Marker::ADD;
  single_match.lifetime = ros::Duration();//(sec,nsec),0 forever


  std::map<__int16_t,std::map<__int16_t, match>>::iterator iter1;
  iter1 = filtered_matches.begin();

  int matches_index = 0;

  for (iter1; iter1 != filtered_matches.end(); ++iter1)
  {
    std::map<__int16_t, match>::iterator iter2;
    iter2 = iter1->second.begin();
    for(iter2; iter2!= iter1->second.end();++iter2)
    {
      if(iter2->second.available)
      {
        single_match.header.stamp=ros::Time::now();
        
        single_match.id= matches_index;         

        geometry_msgs::Point p1, p2;
        p1.x = ins_cens1[iter1->first][iter2->first].x;
        p1.y = ins_cens1[iter1->first][iter2->first].y;
        p1.z = ins_cens1[iter1->first][iter2->first].z;

        p2.x = ins_cens2[iter1->first][iter2->second.id].x;
        p2.y = ins_cens2[iter1->first][iter2->second.id].y;
        p2.z = ins_cens2[iter1->first][iter2->second.id].z + z_offest_;

        single_match.points.push_back(p1);
        single_match.points.push_back(p2);

        output.markers.push_back(single_match);        

        matches_index = matches_index + 1;      
      }
    }
  }

  return output;
}


void TripletGraph::showProgress(float progress)
{
  if (progress >= 1)
	{
		progress = 1;
	}
	int pa = progress * 80;
	std::cout << "\33[1A"; 
	std::cout << "\033[32;1m[" + std::string(pa, '=') + ">" + std::string(80 - pa, ' ') << "]  " << progress * 100 << "%\033[0m" << std::endl;
	fflush(stdout); 
}

results_output TripletGraph::eval(my_pair pair)
{
  //load point cloud with label, and perform pointcloud clustering
  TicToc load_pcs_t;
  pc_results pc_result_1 = get_pointcloud(pair.frame1, 1);
  pc_results pc_result_2 = get_pointcloud(pair.frame2, 2);
  std::cout<<"\033[40;35m[Load point cloud pair and perform clustering] consuming time: \033[0m"<<load_pcs_t.toc()<<"ms"<<std::endl;

  //calculate instance geometric centriods
  TicToc cal_ins_cen_t;
  instance_result ins_result_1 = get_instance_center(pc_result_1.cluster_pc,1);
  instance_result ins_result_2 = get_instance_center(pc_result_2.cluster_pc,2);
  std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens1 = ins_result_1.instance_centriods;
  std::map<__int16_t,std::map<__int16_t, instance_center>> ins_cens2 = ins_result_2.instance_centriods;
  std::cout<<"\033[40;35m[Get instance center] consuming time: \033[0m"<<cal_ins_cen_t.toc()<<"ms"<<std::endl;

  //graph construction
  TicToc build_graph_t;
  Graph_matrixs matrixs_1 = build_graph(ins_cens1);
  Graph_matrixs matrixs_2 = build_graph(ins_cens2);
  std::cout<<"\033[40;35m[Build graphs] consuming time: \033[0m"<<build_graph_t.toc()<<"ms"<<std::endl;

  //descriptor extraction
  TicToc get_descripor_t;
  Descriptors descriptor_1 = get_descriptor(matrixs_1, ins_cens1, ins_result_1.instance_number.second);
  Descriptors descriptor_2 = get_descriptor(matrixs_2, ins_cens2, ins_result_2.instance_number.second);
  std::cout<<"\033[40;35m[Extract descriptor] consuming time: \033[0m"<<get_descripor_t.toc()<<"ms"<<std::endl;

  //vertex matching
  TicToc vertex_match_t;
  std::map<__int16_t,std::map<__int16_t, match>> matches = get_vertex_matches(descriptor_1, descriptor_2);
  std::cout<<"\033[40;35m[Match vertexs] consuming time: \033[0m"<<vertex_match_t.toc()<<"ms"<<std::endl;

  //obtian ground truth for relative pose between pc2 and pc1
  TicToc get_pose_gt_t;
  Eigen::Matrix4d gt_T = get_relative_pose_gt(pair);
  //std::cout<<"\033[40;35m[Get pose gt] consuming time: \033[0m"<<get_pose_gt_t.toc()<<"ms"<<std::endl;

  //frame2 to frame1 6-DoF pose estimation
  TicToc est_pose_t;
  std::pair<Eigen::Matrix4d, Eigen::Matrix4d> est_T_s = pose_estimate(matches, ins_cens1, ins_cens2);
  Eigen::Matrix4d est_T        = est_T_s.second;            //T*
  Eigen::Matrix4d est_T_coarse = est_T_s.first;             //~T
  std::cout<<"\033[40;35m[Get pose estimation] consuming time: \033[0m"<<est_pose_t.toc()<<"ms"<<std::endl;

  //calculate RTE and RRE
  TicToc cal_pose_error_t; 
  std::vector<double> rte_rre = cal_RPE(est_T, gt_T);
  std::cout<<"\033[33;1m[Pose estimation errors]: \033[0m"<<" RTE:"<<rte_rre[0]<<"  , RRE:"<<rte_rre[1]<<"  , RRE1:"<<rte_rre[2]<<std::endl;
  // f_rte_out_<<std::fixed<<std::setprecision(9)<< rte_rre[0]<< std::endl;
  // f_rre_out_<<std::fixed<<std::setprecision(9)<< rte_rre[1]<<" "<<rte_rre[2] <<std::endl;
  //std::cout<<"\033[40;35m[Cal pose error] consuming time: \033[0m"<<cal_pose_error_t.toc()<<"ms"<<std::endl;

  //calculate similarity without projection selection
  TicToc cal_similarity_t; 
  float similarity = cal_similarity(descriptor_1,  descriptor_2);
  std::cout<<"\033[33;1m[Similarity without projection selection]: \033[0m"<<similarity<<std::endl;
  //std::cout<<"\033[40;35m[Cal similarity] consuming time: \033[0m"<<cal_similarity_t.toc()<<"ms"<<std::endl;


  //projection selection
  TicToc filter_match_t;
  std::map<__int16_t,std::map<__int16_t, match>> filtered_matches = select_matches(matches,est_T,ins_cens1,ins_cens2);
  //std::cout<<"\033[40;35m[Filter matches] consuming time: \033[0m"<<filter_match_t.toc()<<"ms"<<std::endl;
  
  //calculate similarity with projection selection
  TicToc refine_similarity_t; 
  float similarity_refined = cal_refined_similarity(descriptor_1,descriptor_2,filtered_matches);
  std::cout<<"\033[33;1m[Similarity with projection selection]: \033[0m"<<similarity_refined<<std::endl;
  //std::cout<<"\033[40;35m[Cal refined similarity] consuming time: \033[0m"<<refine_similarity_t.toc()<<"ms"<<std::endl;  
  std::cout<<std::endl;

  results_output output;
  output.pair_label = pair.label;
  output.instance_number_1 = ins_result_1.instance_number;
  output.instance_number_2 = ins_result_2.instance_number;
  output.similarity = similarity;
  output.similarity_refined = similarity_refined;
  output.rte = rte_rre[0];
  output.rre = rte_rre[1];
  output.rre1= rte_rre[2];

  //visualization
  if(conf_para_.rviz_show)
  {
    //***********clean last frame visualizaiton
    visualization_msgs::Marker marker;
    visualization_msgs::MarkerArray markers;
    marker.id = 0;
    marker.ns = "ins_1";
    marker.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(marker);
    ins_cen1_pub_.publish(markers);

    marker.ns = "ins_2";
    markers.markers.push_back(marker);
    ins_cen2_pub_.publish(markers);

    marker.ns = "edges_graph1";
    markers.markers.push_back(marker);
    edges1_pub_.publish(markers);

    marker.ns = "edges_graph2";
    markers.markers.push_back(marker);
    edges2_pub_.publish(markers);

    marker.ns = "original_matches";
    markers.markers.push_back(marker);
    orignial_matches_pub_.publish(markers);

    marker.ns = "remaining_matches";
    markers.markers.push_back(marker);
    remaining_matches_pub_.publish(markers);
    //***********

    //**********publish
    //pc-1
    sensor_msgs::PointCloud2 pointcloud1;
    pcl::toROSMsg(pc_result_1.original_pc, pointcloud1);           
    pointcloud1.header.frame_id = "tripletgraph";          
    pointcloud1.header.stamp=ros::Time::now();            
    pc1_pub_.publish(pointcloud1);
    
    //pc-2 offest in z
    sensor_msgs::PointCloud2 pointcloud2;
    pcl::PointCloud<pcl::PointXYZRGB> trans_pc2;
    Eigen::Matrix4d offest_T;
    offest_T << 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, z_offest_,
                0.0, 0.0, 0.0, 1.0;
    pcl::transformPointCloud(pc_result_2.original_pc,trans_pc2,offest_T);
    pcl::toROSMsg(trans_pc2, pointcloud2);           
    pointcloud2.header.frame_id = "tripletgraph";          
    pointcloud2.header.stamp=ros::Time::now();            
    pc2_pub_.publish(pointcloud2);    

    //vertex in pc-1
    visualization_msgs::MarkerArray instances_1 = ins_center_visual(ins_cens1, 1);
    ins_cen1_pub_.publish(instances_1);

    //vertex in pc-2
    visualization_msgs::MarkerArray instances_2 = ins_center_visual(ins_cens2, 2);
    ins_cen2_pub_.publish(instances_2);

    //edges
    visualization_msgs::MarkerArray edges_graph1 = edges_visual(ins_cens1, matrixs_1, 1);
    edges1_pub_.publish(edges_graph1);

    visualization_msgs::MarkerArray edges_graph2 = edges_visual(ins_cens2, matrixs_2, 2);
    edges2_pub_.publish(edges_graph2);

    //original matches
    visualization_msgs::MarkerArray original_matches = matches_visual(matches, ins_cens1, ins_cens2);   //20230413
    orignial_matches_pub_.publish(original_matches);
    
    //matches after projection selection operation
    visualization_msgs::MarkerArray matches_selected = remaining_matches_visual(filtered_matches, ins_cens1, ins_cens2);
    remaining_matches_pub_.publish(matches_selected);

    //transform pc-2 to the coordinate system of pc-1, using estimated T*
    pcl::PointCloud<pcl::PointXYZRGB> transformed_pc2; 
    pcl::transformPointCloud(pc_result_2.original_pc, transformed_pc2, est_T);       
    sensor_msgs::PointCloud2 pc2_in_pc1;
    pcl::toROSMsg(transformed_pc2, pc2_in_pc1);         
    pc2_in_pc1.header.frame_id = "tripletgraph";               
    pc2_in_pc1.header.stamp=ros::Time::now(); 
    pc2_in_pc1_pub_.publish(pc2_in_pc1);
  }

  return output;
}