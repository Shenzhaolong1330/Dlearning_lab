"""
shenzhaolong 2025.4.23
This is the controller for quadcopter.
It contains 
    attitude control,
    acceleration control, 
    velocity control,
    position control.

! REMEMBER TO RESET THE CONTROLLER WHEN START AND RESET THE SIMULATION ! use reset_controller()
"""

import torch


class QuadcopterController:
    def __init__(self, num_envs, mass, gravity=9.81, dt=0.005, device="cuda"):
        self.device = device
        # 动力学参数
        # self.mass = torch.full((num_envs,), mass, device=device)
        self.mass = mass
        self.gravity = gravity
        self.base_thrust = self.mass * self.gravity * torch.ones((num_envs,1), device=device)  # 基础推力
        self.dt = dt 
        
        # 姿态控制参数
        self.attitude_gains = torch.tensor([[0.02, 0.02, 0.02]], device=device).repeat(num_envs, 1) # P
        self.angular_rate_gains = torch.tensor([[0.001, 0.001]], device=device).repeat(num_envs, 1) # D
        self.yaw_rate_gain = torch.full((num_envs,), 0.003, device=device) 
        
        # 加速度控制参数
        self.acc_p_gains = torch.ones((num_envs, 3), device=device) * torch.tensor([0.8, 0.8, 0.5], device=device)  # XYZ方向P增益
        self.acc_i_gains = torch.ones((num_envs, 3), device=device) * torch.tensor([40.0, 40.0, 30.0], device=device)  # 积分增益
        self.acc_d_gains = torch.ones((num_envs, 3), device=device) * torch.tensor([0.001, 0.001, 0.0], device=device)  # 微分增益
        self.acc_integral = torch.zeros(num_envs, 3, device=device)  # 积分项
        self.last_acc_error = torch.zeros(num_envs, 3, device=device)  # 上一次加速度误差

        # 速度控制参数
        self.vel_p_gains = torch.ones((num_envs, 3), device=device) * torch.tensor([3.0, 3.0, 3.0], device=device)  # XYZ方向P增益
        self.vel_i_gains = torch.ones((num_envs, 3), device=device) *  torch.tensor([0.04, 0.04, 0.03], device=device)  # 积分增益
        self.vel_d_gains = torch.ones((num_envs, 3), device=device) * 0.001 * torch.tensor([1.0, 1.0, 1.0], device=device) # 微分增益
        self.vel_integral = torch.zeros(num_envs, 3, device=device)  # 积分项
        self.last_vel_error = torch.zeros(num_envs, 3, device=device)  # 上一次速度误差

        # 位置控制参数
        self.pos_p_gains = torch.ones((num_envs, 3), device=device) * torch.tensor([1.0, 1.0, 1.0], device=device)  # XYZ方向P增益

        # 数值稳定性参数
        self.eps = torch.finfo(torch.float32).eps
        self.max_attitude = 0.3

    def reset_controller(self):
        self.vel_integral = torch.zeros_like(self.vel_integral)
        self.last_vel_error = torch.zeros_like(self.last_vel_error)
        self.acc_integral = torch.zeros_like(self.acc_integral)
        self.last_acc_error = torch.zeros_like(self.last_acc_error)

    def gravity_to_attitude(self, projected_gravity):
        # 重力向量在机体坐标系中的投影 (形状: [num_envs, 3])
        g_b = projected_gravity  # 假设该向量已单位化
        
        # 计算滚转角（绕x轴旋转）
        phi = torch.atan2(-g_b[:, 1], -g_b[:, 2])  # 使用y和z分量
        
        # 计算俯仰角（绕y轴旋转）
        theta = torch.atan2(g_b[:, 0], torch.sqrt(g_b[:, 1]**2 + g_b[:, 2]**2 + self.eps))
        
        return torch.stack([phi, theta], dim=1)  # [num_envs, 2]

    def quat_to_euler(self, root_quat_w):
        # 输入: root_quat_w [num_envs, 4] (w, x, y, z)
        w, x, y, z = root_quat_w.unbind(dim=1)
        
        roll = torch.atan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = torch.asin(2*(w*y - z*x))
        yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        
        # output: rad
        return torch.stack([roll, pitch, yaw], dim=1) 
    
    def euler_to_rotation_matrix(self, rpy):
        """
        将欧拉角(roll, pitch, yaw)转换为旋转矩阵（批量处理）
        
        参数：
            rpy : [num_envs, 3] 欧拉角（弧度），顺序为[roll, pitch, yaw]
        
        返回：
            rotation_matrix : [num_envs, 3, 3] 旋转矩阵（机体系→惯性系）
        """
        roll, pitch, yaw = rpy.unbind(dim=1)
        
        # 预计算三角函数值（批量处理）
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        cos_pitch = torch.cos(pitch)
        sin_pitch = torch.sin(pitch)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        # 构建绕X轴的旋转矩阵[4](@ref)
        Rx = torch.zeros((rpy.size(0), 3, 3), device=rpy.device)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1] = cos_roll
        Rx[:, 1, 2] = -sin_roll
        Rx[:, 2, 1] = sin_roll
        Rx[:, 2, 2] = cos_roll
        
        # 构建绕Y轴的旋转矩阵[4](@ref)
        Ry = torch.zeros((rpy.size(0), 3, 3), device=rpy.device)
        Ry[:, 0, 0] = cos_pitch
        Ry[:, 0, 2] = sin_pitch
        Ry[:, 1, 1] = 1.0
        Ry[:, 2, 0] = -sin_pitch
        Ry[:, 2, 2] = cos_pitch
        
        # 构建绕Z轴的旋转矩阵[4](@ref)
        Rz = torch.zeros((rpy.size(0), 3, 3), device=rpy.device)
        Rz[:, 0, 0] = cos_yaw
        Rz[:, 0, 1] = -sin_yaw
        Rz[:, 1, 0] = sin_yaw
        Rz[:, 1, 1] = cos_yaw
        Rz[:, 2, 2] = 1.0
        
        # 组合旋转矩阵（Z-Y-X顺序）[7](@ref)
        rotation_matrix = torch.bmm(Rz, torch.bmm(Ry, Rx))
        return rotation_matrix

    def attitude_controller(self, sensor_data, desired_attitude=None):
        if desired_attitude is None:
            desired_attitude = torch.zeros(sensor_data['base_ang_vel'].shape[0], 3, device=self.device)
        # 输入数据（仅需要角速度和重力投影）
        ang_vel = sensor_data['base_ang_vel'].to(self.device)
        projected_gravity = sensor_data['projected_gravity_b'].to(self.device)
        root_quat_w = sensor_data['root_quat_w'].to(self.device)
        
        # 通过四元数计算姿态角
        rpy = self.quat_to_euler(root_quat_w)
        rpy_error = desired_attitude - rpy
        rp_error = rpy_error[:, :2]
        yaw_error = rpy_error[:, 2]
        # 角速度处理（仅使用roll/pitch轴）
        ang_vel_rp = ang_vel[:, :2]
        
        # PD控制计算
        torque_rp = (
            rp_error * self.attitude_gains[:, :2] +
            (-ang_vel_rp) * self.angular_rate_gains
        )
        
        # 控制器调参时打印
        # print('-'*20)
        # print('姿态角误差:',rpy_error[0])
        # print('-'*20)

        # 推力计算（基于重力投影z分量）
        g_z = torch.clamp(projected_gravity[:, 2], min=-1.0, max=-0.1)  # 重力向下为负
        thrust = self.base_thrust / (-g_z.unsqueeze(1))  # 取反转为正数
        
        # 力矩整合（添加yaw阻尼）
        torque_full = torch.zeros(ang_vel.shape[0], 3, device=self.device)
        torque_full[:, :2] = torque_rp
        torque_full[:, 2] = -ang_vel[:, 2] * self.yaw_rate_gain + yaw_error * self.attitude_gains[:, 2]  # yaw控制
        
        return thrust, torque_full
 
    def acceleration_controller(self, sensor_data, desired_acc=None, desired_yaw=None):
        if desired_acc is None:
            desired_acc = torch.zeros(sensor_data['root_lin_vel_b'].shape[0], 3, device=self.device)
        if desired_yaw is None:
            desired_yaw = torch.zeros(sensor_data['root_lin_vel_b'].shape[0], 1, device=self.device)
        
        # 获取当前姿态的旋转矩阵（机体系→世界系）
        root_quat_w = sensor_data['root_quat_w']
        rotation_matrix = self.euler_to_rotation_matrix(self.quat_to_euler(root_quat_w))
        
        # 将IMU测量的加速度从机体系转换到世界系
        # TODO imu_acc_b 考虑姿态了吗
        imu_acc_b = sensor_data['imu_lin_acc_b']
        imu_acc_w = torch.bmm(rotation_matrix, imu_acc_b.unsqueeze(-1)).squeeze(-1)

        # 计算加速度误差
        acc_error = desired_acc - imu_acc_w
        self.acc_integral += acc_error * self.dt  # 积分项
        derivative = (acc_error - self.last_acc_error) / self.dt  # 微分项
        
        # 计算反馈补偿量（PID控制）
        feedback_acc = (
            acc_error * self.acc_p_gains +
            self.acc_integral * self.acc_i_gains +
            derivative * self.acc_d_gains
        )

        # 世界系下的重力补偿
        gravity_compensation = torch.tensor([0.0, 0.0, -self.gravity], device=self.device)
        total_acc = desired_acc + feedback_acc - gravity_compensation
        
        # 更新误差记录
        self.last_acc_error = acc_error

        # 将期望加速度转换到机体系用于计算推力方向
        desired_acc_b = torch.bmm(rotation_matrix.transpose(1,2), total_acc.unsqueeze(-1)).squeeze(-1)
        thrust_dir = desired_acc_b / (torch.norm(desired_acc_b, dim=1, keepdim=True) + self.eps)
        
        # 计算姿态角（基于机体坐标系下的期望加速度方向）
        roll = - torch.atan2(thrust_dir[:, 1], thrust_dir[:, 2])
        pitch = torch.atan2(thrust_dir[:, 0], thrust_dir[:, 2])
        desired_attitude = torch.stack([roll, pitch, desired_yaw.squeeze(-1)], dim=1)
        
        # 计算总推力（考虑质量）
        thrust_magnitude = torch.norm(total_acc, dim=1) * self.mass
        thrust = thrust_magnitude.unsqueeze(1)
        
        # 调用姿态控制器生成最终力矩
        _, torque = self.attitude_controller(sensor_data, desired_attitude=desired_attitude)
        
        # 计算世界坐标系下的加速度误差 # 控制器调参时打印
        # print('-'*20)
        # print('世界系加速度误差:', acc_error[0])
        # print('加速度误差反馈补偿量',feedback_acc[0])
        # print('-'*20)

        return thrust, torque, desired_attitude

    def velocity_controller(self, sensor_data, desired_vel=None, desired_yaw=None):
        if desired_vel is None:
            desired_vel = torch.zeros(sensor_data['base_ang_vel'].shape[0], 3, device=self.device)
        if desired_yaw is None:
            desired_yaw = torch.zeros(sensor_data['root_lin_vel_b'].shape[0], 1, device=self.device)
        
        # 新增坐标系转换逻辑
        root_quat_w = sensor_data['root_quat_w']
        rotation_matrix = self.euler_to_rotation_matrix(self.quat_to_euler(root_quat_w))
        
        # 将机体速度转换到世界坐标系
        current_vel_b = sensor_data['base_lin_vel']
        current_vel_w = torch.bmm(rotation_matrix, current_vel_b.unsqueeze(-1)).squeeze(-1)

        dt = self.dt
        vel_error = desired_vel - current_vel_w  # 使用世界坐标系速度计算误差

        # ... 保持原有积分和微分计算逻辑 ...
        self.vel_integral += vel_error * dt
        self.vel_integral = torch.clamp(self.vel_integral, -5.0, 5.0)
        derivative = (vel_error - self.last_vel_error) / dt
        self.last_vel_error = vel_error
        
        desired_acc = (
            vel_error * self.vel_p_gains +
            self.vel_integral * self.vel_i_gains +
            derivative * self.vel_d_gains
        )

        thrust, torque, desired_attitude = self.acceleration_controller(
            sensor_data,
            desired_acc=desired_acc,
            desired_yaw=desired_yaw
        )
        
        # 控制器调参时打印
        # print('-'*20)
        # print('世界系速度误差:', vel_error[0])
        # print('-'*20)

        return thrust, torque, desired_attitude, desired_acc

    def position_controller(self, sensor_data, desired_pos=None, desired_yaw=None):
        if desired_pos is None:
            desired_pos = torch.zeros(sensor_data['root_pos_w'].shape[0], 3, device=self.device)
        if desired_yaw is None:
            desired_yaw = torch.zeros(sensor_data['root_lin_vel_b'].shape[0], 1, device=self.device)

        # 获取当前位置
        current_pos = sensor_data['root_pos_w']
        
        # 计算位置误差
        pos_error = desired_pos - current_pos
        
        # 计算期望速度（P控制）
        desired_vel = pos_error * self.pos_p_gains
        
        # 调用速度控制器
        thrust, torque, desired_attitude, desired_acc = self.velocity_controller(
            sensor_data,
            desired_vel=desired_vel,
            desired_yaw=desired_yaw
        )

        # 控制器调参时打印
        print('-'*20)
        print('世界系位置误差:', pos_error[0]) 
        print('-'*20)
        
        return thrust, torque, desired_attitude, desired_acc