//Author:   Weixin Ma       weixin.ma@connect.polyu.hk
#include <ros/ros.h>
#include "./../include/tripletgraph.h"


int main(int argc, char **argv) {
  ros::init(argc, argv, "tripletgraph");
  ros::NodeHandle nh("~");

  TripletGraph node(nh);     

  node.run_single_pair();
  return 0;
}