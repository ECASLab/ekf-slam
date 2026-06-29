"""
Launch MÍNIMO para correr SLAM con la FPGA (robot:=nusim, cmd_src:=circle).

Solo levanta lo que SLAM necesita para funcionar:
  - 2 static_transform_publisher (TF estáticos, quedan idle, costo ~nulo)
  - nusim         (simulador: publica fake_sensor con los landmarks + sensor_data)
  - turtle_control(cmd_vel -> wheel_cmd, sensor_data -> joint_states)
  - circle        (fuente de velocidad)
  - slam          (el nodo con la FPGA, con xclbin_path absoluto)

NO levanta: rviz, odometry, ni los robot_state_publisher / joint_state_publisher
(esos son solo visualización y SLAM no los necesita). Así baja la carga de CPU y
la presión sobre la DDR, SIN tocar ninguna tasa ni velocidad: corre natural.

Uso:
  ros2 launch nuslam slam_fpga.launch.py
  # o con otra ruta de xclbin:
  ros2 launch nuslam slam_fpga.launch.py xclbin_path:=/ruta/a/ekf_spmm.xclbin
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    robot_params = ParameterFile(PathJoinSubstitution(
        [FindPackageShare("nuturtle_description"), "config/diff_params.yaml"]))

    return LaunchDescription([
        DeclareLaunchArgument(
            'xclbin_path',
            default_value='/home/ubuntu/andres.saballo/slam_original_ws/'
                          'ekf_gpa_test/ekf_spmm.xclbin',
            description="ruta absoluta al ekf_spmm.xclbin"),

        # ── TF estáticos (idle tras el arranque, costo ~nulo) ──
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            arguments=["--frame-id", "nusim/world", "--child-frame-id",
                       "blue/odom", "--x", "0.0", "--y", "0.0"]),
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            arguments=["--frame-id", "nusim/world", "--child-frame-id",
                       "map", "--x", "0.0", "--y", "0.0"]),

        # ── Simulador: fake_sensor (landmarks) + sensor_data (encoders) ──
        Node(
            package='nusim', executable='nusim',
            parameters=[
                ParameterFile(PathJoinSubstitution(
                    [FindPackageShare("nusim"), "config/", "basic_world.yaml"])),
                robot_params],
            remappings=[('/red/sensor_data', '/sensor_data'),
                        ('/red/wheel_cmd', '/wheel_cmd')]),

        # ── Control: cmd_vel -> wheel_cmd, sensor_data -> joint_states ──
        Node(
            package='nuturtle_control', executable='turtle_control',
            parameters=[robot_params],
            remappings=[('/joint_states', '/blue/joint_states')]),

        # ── Fuente de velocidad ──
        Node(package='nuturtle_control', executable='circle'),

        # ── SLAM con FPGA ──
        Node(
            package='nuslam', executable='slam',
            parameters=[
                robot_params,
                {"body_id": "green/base_footprint",
                 "odom_id": "green/odom",
                 "wheel_left": "green/wheel_left_joint",
                 "wheel_right": "green/wheel_right_joint",
                 "xclbin_path": LaunchConfiguration('xclbin_path')}],
            remappings=[('/joint_states', '/blue/joint_states')]),
    ])
