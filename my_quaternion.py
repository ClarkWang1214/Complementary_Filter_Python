'''
FilePath: /GitLab/wqc_repo/python/video_stablization_wqc/gyro_vid_stab/src/my_quaternion.py
Description: 四元数相关功能函数
Autor: Clark
Date: 2022-11-30 13:39:20
LastEditors: Clark
LastEditTime: 2022-11-30 13:42:04
Copyright (c) 2022
'''
import numpy as np

'''
brief: 单位四元数的逆，虚部取反
param {*} q 四元数向量[s, x, y, z]
return {*} 
author: Clark
Date: 2022-11-29 09:53:01
'''
def inverse(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])

'''
brief: 四元数的共轭，虚部取反
param {*} q 四元数向量[s, x, y, z]
return {*}
author: Clark
Date: 2022-11-29 09:53:01
'''
def conjugate(q):
    return np.array([q[0],-q[1],-q[2],-q[3]])

'''
brief: 四元数归一化
param {*} q 四元数向量[s, x, y, z]
return {*}
author: Clark
Date: 2022-11-29 08:46:30
'''
def normalize(q):
    return q/np.sqrt(q.dot(q)) # q/|q|

'''
brief: 四元数乘法
param {*} p 第一个四元数
param {*} q 第二个四元数
return {*} 
author: Clark
Date: 2022-11-29 08:58:47
'''
def multiplication(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
                     p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
                     p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
                     p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0])

'''
brief: 过四元数旋转空间向量，获取旋转后的向量
        对应论文<Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs> 公式(9)
param {*} q 旋转四元数
param {*} v 三维向量
return {*}
author: Clark
Date: 2022-11-29 10:00:46
'''
def rotateVectorByQuat1(q, v):
    q0, q1, q2, q3 = q
    vx, vy, vz = v
    # 四元数旋转三维向量，将四元数转换为旋转矩阵，再与三维向量作矩阵乘法
    return np.array([(q0**2 + q1**2 - q2**2 - q3**2) * vx +      2*(q1 * q2 - q0 * q3) * vy      +      2*(q1 * q3 + q0 * q2) * vz, 
                          2*(q1 * q2 + q0 * q3) * vx      + (q0**2 - q1**2 + q2**2 - q3**2) * vy +      2*(q2 * q3 - q0 * q1) * vz, 
                          2*(q1 * q3 - q0 * q2) * vx      +      2*(q2 * q3 + q0 * q1) * vy      + (q0**2 - q1**2 - q2**2 + q3**2) * vz])

'''
brief: 用旋转四元数取旋转某一个三维点坐标向量
param {*} q
param {*} v
return {*}
author: Clark
Date: 2022-10-20 15:57:18
'''
def rotateVectorByQuat2(q, v):
    # 三维向量中用虚四元数[0, x, y, z]表示
    q2 = [0, v[0],v[1],v[2]]
    # 旋转后： p' = qpq^-1
    return multiplication(multiplication(q, q2), conjugate(q))[1:]

'''
brief: 实四元数[1,0,0,0]与旋转四元数[dq0_corr,dq1_corr,dq2_corr,dq3_corr]之间的球面线性插值，互补滤波器需要用
param {*} dq0_corr
param {*} dq1_corr
param {*} dq2_corr
param {*} dq3_corr
param {*} gain
return {*}
author: Clark
Date: 2022-11-28 18:14:38
'''
def slerpScaleQuat(dq, gain):
    if dq[0] < 0.0:
        dq = -dq #取反
    dq0_corr, dq1_corr, dq2_corr, dq3_corr = dq
    # itentity quaternion = [1, 0, 0, 0]^T
    # 与
    # delta quaternion = [dq0, dq1, dq2, dq3]^T
    # 之间进行SLERP/LERP插值

    # 因为四元数向量的余弦值：
    #      cos(theta) = q_I * dq = [1, 0, 0, 0]^T*[dq0, dq1, dq2, dq3]^T = 1*dq0+0*dq1+0*dq2+0*dq3 = dq0
    # 因此，直接看旋转四元数的实部dq0的正负，相当于在检查两个向量夹角的余弦值是否为正
    if (dq0_corr < 0.9995):  # 感觉这个阈值0.0应该是0.9比较合适（论文中就是0.9！），cos(theta)<0.9，两个四元数之间的夹角差不多大于25度
        # 若cos(theta)余弦值为负，说明两个四元数向量之间的夹角大于90度，
        # 则使用Slerp(Spherical linear interpolation)球面线性插值
        # 对应论文<Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs> 公式(52)
        angle = np.arccos(dq0_corr) # 计算两个四元数向量的夹角
        A = np.sin(angle * (1.0 - gain)) / np.sin(angle) # 计算系数A
        B = np.sin(angle * gain) / np.sin(angle) #计算系数B

        # SLERP公式，由于左边的四元数为[1,0,0,0]^T，因此对应元素相乘相加即可
        #     q' = A*q_I+B*dq
        dq0_corr = A + B * dq0_corr
        dq1_corr = B * dq1_corr
        dq2_corr = B * dq2_corr
        dq3_corr = B * dq3_corr
    else:
        # 若余弦值大于0.9，则说明两个四元数向量之间的夹角小于25度的时候，直接用简单的线性插值
        # 则使用线性插值Lerp (Linear interpolation)，对应论文公式(50)
        dq0_corr = (1.0 - gain) + gain * dq0_corr
        dq1_corr = gain * dq1_corr
        dq2_corr = gain * dq2_corr
        dq3_corr = gain * dq3_corr
    
    # 归一化为单位四元数
    return normalize(np.array([dq0_corr, dq1_corr, dq2_corr, dq3_corr]))

'''
brief: 【标准SLERP】球面线性插值得到时间戳t处的四元数值
param {*} q0   t0时刻的四元数
param {*} q1   t1时刻的四元数
param {*} alpha    比例系数
return {*} 返回插值后的四元数
author: Clark
Date: 2022-08-16 17:44:05
'''
def slerp(q0, q1, alpha):
    q0 = np.array(q0) 
    q1 = np.array(q1)
    dot = q0.dot(q1) # cos(theta) = q0*q1
    
    # 若两个四元数各对应元素乘积之和小于零
    if dot < 0.0:
        q1 = -q1 #取反
        dot = -dot #取反
    
    # 两个四元数的余弦值是否大于阈值
    if dot > 0.9995:
        # 如果dot大于最大阈值，说明p和q之间的夹角很小，sin(theta) -> 0，此时除法会有问题，可使用简单的线性插值替代
        # 退化为：slerp(q0, q1, alpha) = (1-alpha)*q0 + alpha*q1，参考：https://zhuanlan.zhihu.com/p/538653027
        result = (1-alpha)*q0 + alpha*q1
        return result / np.linalg.norm(result) # 插值后的四元数归一化为单位四元数
    
    angle = np.arccos(dot) # 计算两个四元数向量的夹角
    A = np.sin(angle * (1.0 - alpha)) / np.sin(angle) # 计算系数A
    B = np.sin(angle * alpha) / np.sin(angle) #计算系数B

    return normalize(A*q0 + B*q1) # 归一化为单位四元数

'''
brief: 计算两个四元数之间的旋转
param {*} q1 前一个朝向四元数
param {*} q2 后一个朝向四元数
return {*} 两个朝向四元数之间的旋转四元数
author: Clark
Date: 2022-08-16 17:31:58
'''
def rot_quat_between(q1, q2):
    return multiplication(inverse(q1), q2)

'''
brief: 两个朝向四元数之间的旋转角度，向量余弦值
param {*} q1 前一个朝向四元数
param {*} q2 后一个朝向四元数
return {*} 两个朝向四元数之间的旋转角度
author: Clark
Date: 2022-11-29 09:12:17
'''
def angle_between(q1, q2):
    # 两个朝向四元数之间的旋转四元数： q = q1^-1 * q2
    rot_quat = rot_quat_between(q1, q2)
    # cos(theta) = 
    theta = 2 * np.arccos(min(rot_quat[0], 1))
    return theta

# 测试函数
if __name__ == "__main__":
    # 四元数
    q = np.array([0.5,0.7,0.5,0.5])

    # 三维向量
    v = np.array([1,2,3])

    # 用四元数旋转三维向量，两种实现
    v_rotated1 = rotateVectorByQuat1(q,v)
    v_rotated2 = rotateVectorByQuat2(q,v)
    
    # 四元数球面线性插值
    a = np.array([1, 0, 0, 0])
    b = np.array([0.5,0.7,0.5,0.5])
    c = slerp(a, b, 0.6) 
    d = slerpScaleQuat(b, 0.6) 

    exit()

