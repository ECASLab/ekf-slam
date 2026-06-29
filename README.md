# ekf-slam
ekf-slam

Instructions to compile and run EKF-SLAM


A) Original EKF-SLAM

1. Go to: slam_original_ws/src/nuslam/src and change slam.cpp content with slam_CPU.cpp
	  slam_original_ws/src/nuslam and change CMakeLists.txt content with CMakeLists_CPU.txt

2. Go to: slam_original_ws/ and load the environment:
	source /opt/ros/humble/setup.bash
	source ~/slam_original_ws/install/setup.bash
	
3. Compile using: colcon build --symlink-install --packages-select nuslam

4. Run using: ros2 launch nuslam nuslam.launch.py \ robot:=nusim cmd_src:=circle use_rviz:=false
	This command will launch the SLAM without showing the ROS2 visual environment

   	If you want to run with visual environment use: ros2 launch nuslam unknown_data_assoc.launch.py
   
   
========================================================================================================================


B) FPGA EKF-SLAM

1. Go to: slam_original_ws/src/nuslam/src and change slam.cpp content with slam_FPGA.cpp
	  slam_original_ws/src/nuslam and change CMakeLists.txt content with CMakeLists_FPGA.txt

2. Go to: slam_original_ws/ and load the environment:
	source /opt/ros/humble/setup.bash
	source ~/slam_original_ws/install/setup.bash

3. Use command: sudo xmutil unloadapp; sudo xmutil loadapp ekf_spmm

4. Compile using: MAKEFLAGS=-j1 colcon build --symlink-install --packages-select nuslam --parallel-workers 1

5. Run using: ros2 launch nuslam slam_fpga.launch.py
	This command will launch the SLAM without showing the ROS2 visual environment and sending data through the FPGA.
