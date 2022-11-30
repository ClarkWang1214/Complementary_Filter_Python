'''
Description: 互补滤波积分，使用陀螺仪角速度+加速度数据，计算IMU朝向四元数

参考论文：《Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs》

Autor: Clark 
Date: 2022-11-30 13:26:28
LastEditors: WQC
LastEditTime: 2022-11-30 13:44:12
Copyright: Copyright (c) 2021
'''

import os
import numpy as np
import my_quaternion

'''
brief: 互补滤波积分器，融合陀螺仪与加速度计得到IMU朝向四元数
author: Clark
Date: 2022-11-28 18:02:51
'''
class ComplementaryIntegrator:
    '''
    brief: 构造函数初始化
    param {*} self
    author: Clark
    Date: 2022-11-28 18:05:09
    '''    
    def __init__(self):
        # 朝向四元数是否初始化
        self.initialized = False 
        # 当前IMU是否为稳态标记
        self.steady_state = False 
        # 是否估计陀螺仪零偏
        self.do_bias_estimation = True 
        # 是否计算自适应增益
        self.do_adaptive_gain = True 

        # 重力加速度
        self.kGravity = 9.81 
        # 常量增益值
        self.gain_acc = 0.0004 
        # 陀螺零偏估计alpha值
        self.bias_alpha = 0.001 
        # 加速度阈值
        self.kAccelerationThreshold = 0.1 
        # 角速度阈值
        self.kAngularVelocityThreshold = 0.2 
        # 角速度变化阈值
        self.kDeltaAngularVelocityThreshold = 0.01 

        # global -> local 全局坐标系相对于局部坐标系朝向四元数
        self.orientation = np.array([1.0,0.0,0.0,0.0])
        # 陀螺仪预测的当前时刻IMU的朝向四元数
        self.q_predict = np.array([1.0,0.0,0.0,0.0]) 
        # 加速度计得到的旋转四元数
        self.delta_q = np.array([1.0,0.0,0.0,0.0])
        # 前一个陀螺仪角速度
        self.gyro_prev = np.array([1.0,0.0,0.0])
        # 陀螺仪零偏
        self.gyro_bias = np.array([0.0,0.0,0.0])

    '''
    brief: 互补滤波器估计当前时刻IMU的朝向四元数
    param {*} self
    param {*} gyr_data 陀螺仪读数
    param {*} acc_data 加速度计读数
    param {*} dt 前后IMU的时间间隔
    author: Clark
    Date: 2022-11-28 16:23:34
    '''    
    def update(self, gyr_data, acc_data, dt):
        # 朝向四元数是否初始化
        if not self.initialized:
            # 若未初始化，采用加速度计读数【初始化朝向四元数】
            self.initialize(acc_data)
            # 状态变为已初始化
            self.initialized = True
            return # 直接返回
        
        # 【陀螺仪零偏估计】
        if self.do_bias_estimation:
            self.updateBias(gyr_data, acc_data)

        # 【预测Prediction】
        # 陀螺仪角速度估计dt时间间隔后的朝向四元数
        self.q_predict = self.predict(gyr_data, dt)

        # 【修正Correction】
        # 根据加速度计读数与陀螺预测的朝向四元数
        # 计算修正用的旋转四元数delta quaternion
        self.delta_q = self.correct(acc_data,       # 加速度计读数
                                    self.q_predict) # 陀螺预测的朝向四元数

        # 【自适应增益Adaptive Gain】
        if self.do_adaptive_gain:
            adaptGain = self.computeAdaptiveGain(acc_data)
        else:
            adaptGain = self.gain_acc

        # 利用计算得到的自适应增益值，对旋转四元数进行【球面线性插值SLERP】
        self.delta_q = my_quaternion.slerpScaleQuat(self.delta_q, # 加速度计算的旋转四元数
                                                    adaptGain)    # 自适应增益

        # 朝向四元数与旋转四元数作四元数乘法，得到修正后的下一时刻朝向四元数
        self.orientation = my_quaternion.multiplication(self.q_predict, # 角速度预测的朝向四元数
                                                        self.delta_q)   # 加速度计算的旋转四元数

        # 归一化为单位四元数
        self.orientation = my_quaternion.normalize(self.orientation)

    '''
    brief: 获取朝向四元数(local->global)
    param {*} self
    return {*}
    author: Clark
    Date: 2022-11-28 18:50:43
    '''    
    def getOrientation(self):
        # 取逆，得到(local->global)
        return my_quaternion.inverse(self.orientation) # [q0_, q1_, q2_, q3_]: global->local  

    '''
    brief: 加速度计读数初始化朝向四元数
    param {*} self
    param {*} acc_data 加速度计读数
    author: Clark
    Date: 2022-11-28 16:32:20
    '''    
    def initialize(self, acc_data):
        # 归一化为单位向量 a/|a|
        acc_data = acc_data/np.linalg.norm(acc_data)
        ax, ay, az = acc_data

        # 根据z轴重力分量的正负计算当前姿态的四元数，要求IMU安装方向要正确尽量不要出现45度仰望天空
        # 对应论文<Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs> 第4节公式（25）
        if az>=0:
            tmp = np.sqrt((az+1)*0.5)
            self.orientation[0] = tmp
            self.orientation[1] = -ay/(2.0*tmp)
            self.orientation[2] = ax/(2.0*tmp)
            self.orientation[3] = 0
        else:
            tmp = np.sqrt((1-az)*0.5)
            self.orientation[0] = -ay/(2.0*tmp)
            self.orientation[1] = tmp
            self.orientation[2] = 0
            self.orientation[3] = ax/(2.0*tmp)

    '''
    brief: 更新陀螺仪零偏
    param {*} self
    param {*} gyr_data 角速度读数
    param {*} acc_data 加速度计读数
    author: Clark
    Date: 2022-11-28 16:47:39
    '''    
    def updateBias(self, gyr_data, acc_data):
        # 检查IMU是否处于稳态(steady-state)
        self.steady_state = self.checkImuState(gyr_data, acc_data)
        if self.steady_state:
            # 如果传感器处于稳态
            # 执行low-pass filter低通滤波器更新陀螺仪的零偏bias，原始陀螺仪读数减去估计的零偏bias
            self.gyro_bias = self.gyro_bias + self.bias_alpha*(gyr_data - self.gyro_bias)
        
        # 更新当前角速度为上一个
        self.gyro_prev = gyr_data

    '''
    brief: 检查当前IMU是否处于稳态
    param {*} self
    param {*} gyr_data 角速度读数
    param {*} acc_data 加速度计读数
    return {*}
    author: Clark
    Date: 2022-11-28 17:03:49
    '''    
    def checkImuState(self, gyr_data, acc_data):
        # 加速度计读数模长
        acc_magnitude = np.linalg.norm(acc_data)

        # 减去重力加速度后是否超过加速度计阈值9.81
        if (np.fabs(acc_magnitude - self.kGravity) > self.kAccelerationThreshold):
            return False

        # 当前陀螺仪角速度减去上一个陀螺仪角速度的差值，是否超过角速度变化阈值
        if np.sum(np.fabs(gyr_data - self.gyro_prev) > self.kDeltaAngularVelocityThreshold) > 0:
            return False

        # 当前陀螺仪角速度减去零偏后，是否超过角速度阈值
        if np.sum(np.fabs(gyr_data - self.gyro_bias) > self.kAngularVelocityThreshold) > 0:
            return False

        return True

    '''
    brief: 陀螺仪角速度积分得到朝向四元数，对应论文公式(42)
    param {*} self
    param {*} gyr_data 陀螺仪角速度
    param {*} dt 时间间隔
    return {*}
    author: Clark
    Date: 2022-11-28 17:30:39
    '''    
    def predict(self, gyr_data, dt):
        # 原始角速度减去陀螺仪零偏
        gyr_data_unbias = gyr_data - self.gyro_bias
        gx_unbias, gy_unbias, gz_unbias = gyr_data_unbias

        # 对应论文5.1小节 Prediction
        # 公式（39）、（40）、（41）、（42）
        #         |wx|               |  0 -wz  wy|             |q0|
        # omega = |wy|   |omega|_x = | wz   0 -wx|   {^L_G}q = |q1|    global -> local（传感器测量得到的是：local -> global）
        #         |wz|               |-wy  wx   0|             |q2|
        #                                                      |q3|
        # \Omega(^L\omega_{t_k}) = |   0      omega^T  | 
        #                          |-omega  -|omega|_x | 
        # 
        # {^L_G} \dot{q}_{omega, t_k} = \Omega(^L\omega_{t_k}) * {^L_G}q_{t_{k-1}} 
        #                             = |  0  wx  wy  wz|   |q0|
        #                               |-wx   0  wz -wy| * |q1|
        #                               |-wy -wz   0  wx|   |q2|
        #                               |-wz  wy -wx   0|   |q3|
        #                             = | wx*q1 + wy*q2 + wz*q3|
        #                               |-wx*q0 + wz*q2 - wy*q3|
        #                               |-wy*q0 - wz*q1 + wx*q3|
        #                               |-wz*q0 + wy*q1 - wx*q2|

        #  q_{omega,t_k} = q_{omega,t_{k-1}} + 0.5*dt*\Omega(omega_{t_k})*q_{omega,t_{k-1}} 
        #                  |q0|          |  0  wx  wy  wz| |q0|   |q0|          | wxq1+wyq2+wzq3|
        #                = |q1| + 0.5*dt*|-wx   0  wz -wy|*|q1| = |q1| + 0.5*dt*|-wxq0+wzq2-wyq3|
        #                  |q2|          |-wy -wz   0  wx| |q2|   |q2|          |-wyq0-wzq1+wxq3|
        #                  |q3|          |-wz  wy -wx   0| |q3|   |q3|          |-wzq0+wyq1-wxq2|

        # 对应于公式（42）陀螺仪角速度估计的初始朝向四元数
        # q0_,q1_,q2_,q3_是前一时刻全局坐标系相对于局部坐标系的朝向四元数
        # 预测得到当前dt时间间隔之后朝向四元数
        q0_, q1_, q2_, q3_ = self.orientation
        return np.array([q0_ + 0.5 * dt * (         0 * q0_ + gx_unbias * q1_ + gy_unbias * q2_ + gz_unbias * q3_),
                         q1_ + 0.5 * dt * (-gx_unbias * q0_ +         0 * q1_ + gz_unbias * q2_ - gy_unbias * q3_),
                         q2_ + 0.5 * dt * (-gy_unbias * q0_ - gz_unbias * q1_ +         0 * q2_ + gx_unbias * q3_),
                         q3_ + 0.5 * dt * (-gz_unbias * q0_ + gy_unbias * q1_ - gx_unbias * q2_ +         0 * q3_)])

    '''
    brief: 根据加速度计和陀螺仪预测的朝向四元数，计算修正用的旋转四元数
    param {*} self
    param {*} acc_data 加速度计读数
    param {*} q_pred 陀螺仪角速度预测的朝向四元数(global->local)
    return {*} 修正用的旋转四元数 delta quaternion
    author: Clark
    Date: 2022-11-28 17:49:21
    '''    
    def correct(self, acc_data, q_pred):
        # 归一化加速度计读数为单位向量 a/|a|
        acc_data = acc_data/np.linalg.norm(acc_data)

        # 用陀螺仪预测得到的朝向四元数(global->local)的逆(local->global)，虚部取反[p0, -p1, -p2, -p3]^T
        # 将加速度计读数向量旋转到全局坐标系下，对应论文公式(44)
        acc_pred = my_quaternion.rotateVectorByQuat1(my_quaternion.inverse(q_pred), # 取逆，得到 local->global
                                                     acc_data) # 加速度计读数
        ax, ay, az = acc_pred
        # Delta quaternion that rotates "the real gravity [0 0 1]^T" into "the predicted gravity [gx gy gz]^T":
        # R(\Delta q_{acc})*[0 0 1]^T = [gx gy gz]^ 对应论文公式（45）
        # 公式(47)是公式(45)的闭式解，得到修正四元数dq
        tmp = np.sqrt((az+1)*0.5)
        return np.array([ tmp,
                         -ay/(2.0*tmp),
                          ax/(2.0*tmp),
                          0.0]) # 令 \Delta q_{3_{acc}]} = 0，论文公式（19）的方程组才会有解

    '''
    brief: 计算自适应增益，用于缩放旋转四元数delta quaternion
    param {*} self
    param {*} acc_data 加速度计读数
    return {*}
    author: Clark
    Date: 2022-11-28 18:36:11
    '''    
    def computeAdaptiveGain(self, acc_data):
        # 对应论文《Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs》 公式(60)(61)
        a_mag = np.linalg.norm(acc_data) # 加速度读数向量幅值模长
        error = np.fabs(a_mag - self.kGravity) / self.kGravity # 幅值误差，对应论文公式(60)
        error1 = 0.1 # 第一个阈值
        error2 = 0.2 # 第二个阈值
        m = 1.0 / (error1 - error2) # 直线方程的斜率：m = (y1-y0)/(x1-x0) = -10.0
        b = 1.0 - m * error1 # 2.0
        if error < error1:
            factor = 1.0 # 小于0.1的，增益因子为1
        elif error < error2:
            factor = m * error + b # 大于0.1且小于0.2的，增益因子是一个分段连续函数，线性递减
        else:
            factor = 0.0 # 大于0.2的，增益因子降为0
        return factor * self.bias_alpha # 对应论文公式(61)


# 测试函数
if __name__ == "__main__":
    root = '/home/clark/hgfs/Projects/data_record/2022-10-26'

    # 加载IMU时间戳与读数
    imu_data = np.loadtxt(os.path.join(root, 'imu_data.txt'))

    # 初始化互补滤波器
    integrator = ComplementaryIntegrator()
    
    is_first_imu = True # 是否首帧标记
    orientation_list = [] # 朝向四元数列表
    ts_list = [] # 时间戳列表

    # 遍历所有IMU数据，执行互补滤波器，积分得到IMU的朝向四元数
    for i in range(len(imu_data)):
        tmp = imu_data[i,:]
        ts = tmp[0]/1000 # 时间戳，单位秒
        gyr = tmp[1:4] # 角速度
        acc = tmp[4:7] # 加速度

        # 是否为第一个IMU数据
        if is_first_imu:
            ts_prev = ts
            is_first_imu = False
            continue # 跳过

        # 计算前后IMU时间间隔
        dt = ts - ts_prev
        
        # 执行互补滤波器，根据陀螺与加速度计读数估计当前IMU的朝向四元数
        integrator.update(gyr, acc, dt)

        # 获取dt时间后的朝向四元数，局部坐标系->全局坐标系
        orientation = integrator.getOrientation()

        # 添加当前IMU的朝向四元数
        orientation_list.append(orientation)
        # 添加当前IMU的时间戳
        ts_list.append(ts)

        # 更新时间戳
        ts_prev = ts 
    
    exit()