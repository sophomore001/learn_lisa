import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
from scipy.linalg import lstsq
from scipy.signal import fftconvolve
from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import lsq_linear

# 设置显示中文字体（适用于Windows系统）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 设置正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False


# ====================================================
# 论文章节：CLOCK ERRORS MODELLING & SIMULATION OF CLOCK ERRORS
# 实现时钟噪声生成器核心函数
# ====================================================
class ClockErrorGenerator:
    # 论文中按照tau_s=1.0仿真，实际程序中tau_s为变量
    # def __init__(self, tau_s=1.0, f_h=1.0):
    #     """
    #     初始化时钟误差生成器
    #     :param tau_s: 采样间隔时间（单位：秒）
    #     :param f_h: 高频截止频率（单位：Hz）
    #     """
    #     self.tau_s = tau_s  # 采样间隔（秒）
    #     self.f_h = f_h  # 高频截止频率
    def __init__(self, tau_s, f_h=1.0):
        """
        初始化时钟误差生成器
        :param tau_s: 采样间隔时间（单位：秒）
        :param f_h: 高频截止频率（单位：Hz）
        """
        self.tau_s = tau_s  # 采样间隔（秒）
        self.f_h = f_h  # 高频截止频率

    def white_phase_noise(self, A_wp, N):
        """
        生成白相位噪声（White Phase Noise）
        :param A_wp: 白相位噪声幅度系数
        :param N: 生成噪声序列长度
        :return: 白相位噪声序列（一维数组）
        """
        rand = np.random.uniform(-1, 1, N)
        # 通过差分运算生成高频噪声，符合白相位噪声的功率谱特性
        y = A_wp * np.sqrt(self.f_h / self.tau_s ** 2) * (rand[1:] - rand[:-1])
        z = np.pad(y, (1, 0), 'constant')
        return z  # 对齐时间索引

    def white_frequency_noise(self, A_wf, N):
        """
        生成白频率噪声（White Frequency Noise）
        :param A_wf: 白频率噪声幅度系数
        :param N: 生成噪声序列长度
        :return: 白频率噪声序列（一维数组）
        """
        rand = np.random.uniform(-1, 1, N)
        # 直接使用均匀分布生成，sqrt(3)用于调整均匀分布方差
        y = A_wf * np.sqrt(1 / self.tau_s) * np.sqrt(3) * rand
        return y

    def random_walk_noise(self, A_rw, N):
        """
        生成随机游走噪声（Random Walk Noise）
        :param A_rw: 随机游走噪声幅度系数
        :param N: 生成噪声序列长度
        :return: 随机游走噪声序列（一维数组）
        """
        rand = np.random.uniform(-1, 1, N)
        # 通过累积求和生成低频噪声，符合随机游走特性
        y = np.cumsum(A_rw * np.sqrt(self.tau_s) * 3 * rand)
        return y

    def flicker_frequency_noise(self, A_ff, N):
        """
        生成闪烁频率噪声（Flicker Frequency Noise，1/f噪声）
        使用FFT卷积法生成精确的1/f噪声
        :param A_ff: 闪烁噪声幅度系数
        :param N: 生成噪声序列长度
        :return: 闪烁频率噪声序列（一维数组）
        """
        # 计算最接近的2的幂次长度以提高FFT效率
        n = int(np.log2(N)) + 1
        M = 2 ** n
        # 构建功率谱衰减权重（符合1/f特性）
        weights = np.arange(1, M + 1) ** (-2 / 3)
        weights /= np.sqrt(np.sum(weights ** 2))  # 归一化
        # 生成白噪声并进行频谱卷积
        white_noise = np.random.uniform(-1, 1, M)
        conv = fftconvolve(white_noise, weights, mode='full')[:M]
        # 调整幅度和方差以匹配理论值
        return A_ff * np.sqrt(5) * conv[:N] * np.sqrt(12)  # 调整均匀分布方差

    def generate_clock_errors(self, params, N):
        """
        生成综合时钟误差（时间偏差序列）
        :param params: 噪声参数字典，包含A_wp, A_wf, A_ff, A_rw四个键, A对应不同噪声类型的一秒Allan偏差
        :param N: 生成序列长度
        :return: (总时间偏差序列, 各噪声分量字典)
        """
        # 生成各噪声分量
        components = {
            'wp': self.white_phase_noise(params['A_wp'], N),
            'wf': self.white_frequency_noise(params['A_wf'], N),
            'ff': self.flicker_frequency_noise(params['A_ff'], N),
            'rw': self.random_walk_noise(params['A_rw'], N)
        }
        # 叠加频率噪声分量并进行时间积分得到时间偏差
        y_total = sum(components.values())
        x = np.cumsum(y_total) * self.tau_s  # 积分公式：x(t) = ∫y(t)dt
        return y_total, x, components  # 四种噪声分量叠加是y_total,也是频率偏差的叠加
        # # 只使用闪烁频率噪声分量
        # y_total = components['ff']  # 直接使用闪烁频率噪声分量
        # x = np.cumsum(y_total) * self.tau_s  # 积分公式：x(t) = ∫y(t)dt
        # return y_total, x, components  # 返回频率偏差、时间偏差和所有噪声分量


# ====================================================
# 论文章节：SIMULATION RESULTS
# 实现Allan方差计算和可视化
# ====================================================
def linear_initial_guess_squared(taus, adev_measured, min_val=1e-16):
    """
    对 P = sigma_y^2 做线性最小二乘拟合：
      P = h0*(1/τ^2) + h1*(1/τ) + h2*(1) + h3*(τ)
    解出 h >= 0 后 A = sqrt(h) 作为初始猜测
    """
    P = adev_measured**2
    X = np.column_stack([
        1 / taus**2,    # 对应 A_wp^2
        1 / taus,       # 对应 A_wf^2
        np.ones_like(taus),  # 对应 A_ff^2
        taus            # 对应 A_rw^2
    ])
    res = lsq_linear(X, P, bounds=(0, np.inf), lsmr_tol='auto')
    h = res.x
    return np.sqrt(np.maximum(h, min_val))

def estimate_noise_parameters(taus, adev_measured,
                              lm_eps=1e-8, lm_max_iter=100, lm_lambda_init=1e-3):
    """
    基于 Allan 偏差测量值的噪声参数估计，
    先线性最小二乘得初值，再 LM-logscale 阻尼牛顿精调。
    返回 [A_wp, A_wf, A_ff, A_rw]
    """
    # 1. 线性初始猜测
    A0 = linear_initial_guess_squared(taus, adev_measured)
    Y = np.log(A0)  # LM-logscale: 对数域参数
    lam = lm_lambda_init

    # 2. LM-logscale 阻尼牛顿迭代
    for _ in range(lm_max_iter):
        A = np.exp(Y)
        # 模型预测和残差
        P = (A[0]/taus)**2 + (A[1]/np.sqrt(taus))**2 + A[2]**2 + (A[3]*np.sqrt(taus))**2
        sqrtP = np.sqrt(P)
        F = sqrtP - adev_measured

        # 雅可比 J_A 对 A，再链式求 J_Y = J_A * dA/dY = J_A * A
        J_A = np.vstack([
            A[0]/(taus ** 2 * sqrtP),
            A[1]/(taus * sqrtP),
            A[2] / sqrtP,
            A[3]*taus / sqrtP
        ]).T
        J_Y = J_A * A

        # LM Hessian 修正： H = J^T J + λ diag(J^T J)
        JTJ = J_Y.T @ J_Y
        H = JTJ + lam * np.diag(np.diag(JTJ))
        g = J_Y.T @ F

        # 求 dY
        try:
            dY = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dY = -np.linalg.lstsq(H, g, rcond=None)[0]

        # 测试更新
        Y_new = Y + dY
        A_new = np.exp(Y_new)
        P_new = (A_new[0]/taus)**2 + (A_new[1]/np.sqrt(taus))**2 + A_new[2]**2 + (A_new[3]*np.sqrt(taus))**2
        F_new = np.sqrt(P_new) - adev_measured

        # 接受或拒绝，并动态调整 λ
        if np.linalg.norm(F_new) < np.linalg.norm(F):
            Y, lam = Y_new, lam * 0.5
        else:
            lam *= 2.0

        # 收敛判定
        if np.linalg.norm(dY) < lm_eps:
            break

    return np.exp(Y)



# def linear_initial_guess(taus, adev_measured, min_val=1e-16):
#     """使用线性最小二乘法估计初始参数"""
#     X = np.column_stack([np.sqrt(1 / taus ** 2),
#                          np.sqrt(1 / taus),
#                          np.ones_like(taus),
#                          np.sqrt(taus)])
#     y = adev_measured
#     res = lsq_linear(X, y, bounds=(0, np.inf))
#     h = res.x  # res是字典，x是一维数组
#     A_initial = np.maximum(h, min_val)  # 防止负数
#     print('A_initial: ', A_initial)
#     return A_initial
#
#
# def estimate_noise_parameters(taus, adev_measured):
#     """
#     基于Allan方差测量值的噪声参数估计，使用改进的阻尼牛顿迭代法
#     :param taus: 积分时间数组
#     :param adev_measured: 测量的Allan标准差数组
#     :return: 估计的噪声参数字典
#     """
#     # 使用线性最小二乘法获取初始参数估计
#     A_initial = linear_initial_guess(taus, adev_measured)
#     X0 = A_initial.copy()
#     X = np.array(X0, dtype=float)
#     A_wp, A_wf, A_ff, A_rw = X
#
#     # 迭代参数设置
#     max_iter = 100  # 最大迭代次数
#     tol = 1e-15  # 收敛阈值
#     for k in range(max_iter):
#         P = (A_wp / taus) ** 2 + (A_wf / taus ** (1 / 2)) ** 2 + A_ff ** 2 + (A_rw * taus ** (1 / 2)) ** 2
#         adev_predicted = np.sqrt(P)  # 预测的Allan标准差
#         F = adev_predicted - adev_measured  # 残差
#
#         # 计算雅可比矩阵
#         J = np.zeros((len(taus), 4))
#         sqrt_P = np.sqrt(P)
#         J[:, 0] = A_wp / (taus ** 2 * sqrt_P)  # 白相位噪声项的偏导数
#         J[:, 1] = A_wf / (taus * sqrt_P)  # 白频率噪声项的偏导数
#         J[:, 2] = A_ff / sqrt_P  # 闪烁频率噪声项的偏导数
#         J[:, 3] = A_rw * taus / sqrt_P  # 随机游走噪声项的偏导数
#
#         # 计算 Gauss-Newton 增量 ΔX = -(J^T J)^{-1} J^T F
#         # 等价于最小二乘解 J ΔX = -F
#         # 使用 np.linalg.lstsq 求最小二乘解
#         # 或显式求解正则方程: JtJ ΔX = -J^T F
#         JtJ = J.T @ J
#         JtF = J.T @ F
#         try:
#             deltaX = -np.linalg.solve(JtJ, JtF)
#         except np.linalg.LinAlgError:
#             # J^T J 奇异时退化为最小二乘求解
#             deltaX, *_ = np.linalg.lstsq(JtJ, -JtF, rcond=None)
#
#         # 收敛判断
#         norm_dx = np.linalg.norm(deltaX)
#         if norm_dx < tol:
#             print(f"Iter{k} : Converged with ΔX norm = {norm_dx:.2e}")
#             break
#
#         # 阻尼线搜索并强制非负
#         alpha = 1.0
#         cost_old = np.sum(F ** 2)
#         min_val = 1e-16  # 最小值限制
#         while alpha > 1e-6:
#             X_new = X + alpha * deltaX
#             X_new = np.maximum(X_new, min_val)
#             P_new = (X_new[0] / taus) ** 2 + (X_new[1] / np.sqrt(taus)) ** 2 + X_new[2] ** 2 + (
#                         X_new[3] * np.sqrt(taus)) ** 2
#             F_new = np.sqrt(P_new) - adev_measured
#             if np.sum(F_new ** 2) < cost_old:
#                 X = X_new
#                 break
#             alpha *= 0.5
#         else:
#             # 依然更新但确保非负
#             X = np.maximum(X + deltaX, min_val)
#
#     return X

    # # Levenberg-Marquardt迭代优化
    # # 牛顿迭代法
    # prev_cost = np.inf
    # for i in range(max_iter):
    #     # r = residual(theta)
    #     J = jacobian(theta, taus)
    #     G = J.T @ J
    #     delata_Xk = np.linalg.inv(G) @ J.T @ model_adev(theta)
    #     theta_new = theta - delata_Xk
    #     J = jacobian(theta_new, taus)


#         # 计算Levenberg-Marquardt阻尼项
#         lambda_lm = 1e-4 * np.trace(J.T @ J) / 4
#         delta = np.linalg.solve(J.T @ J + lambda_lm * np.eye(4), -J.T @ r)
#
#         # 回溯线搜索以确保收敛
#         alpha = 1.0
#         for _ in range(20):
#             theta_new = theta + alpha * delta
#             theta_new = np.maximum(theta_new, 1e-20)  # 确保参数非负
#             r_new = residual(theta_new)
#             cost_new = np.sum(r_new ** 2)
#
#             if cost_new < prev_cost:
#                 theta = theta_new
#                 prev_cost = cost_new
#                 break
#             alpha *= 0.5  # 减小步长
#
#         # 检查收敛条件
#         if np.linalg.norm(alpha * delta) < tol:
#             break
#
#     # 返回估计的噪声参数
#     estimated_params = {
#         'A_wp': theta[0],  # 白相位噪声系数
#         'A_wf': theta[1],  # 白频率噪声系数
#         'A_ff': theta[2],  # 闪烁频率噪声系数
#         'A_rw': theta[3]   # 随机游走噪声系数
#     }
#     return estimated_params


# def estimate_noise_parameters(taus, adev_measured):
#     """
#     基于Allan方差测量值的噪声参数估计
#     :param taus: 积分时间数组
#     :param adev_measured: 测量的Allan标准差数组
#     :return: 估计的噪声参数字典（包含A_wp, A_wf, A_ff, A_rw）
#     """
#     fh = 1  # 高频截止频率
#
#     # 构建设计矩阵 (根据Allan方差理论模型)
#     # H 5x4
#     H = np.vstack([
#         fh / np.array(taus) ** 2,  # 白相位噪声项 (1/τ²)
#         1 / np.array(taus),  # 白频率噪声项 (1/τ)
#         np.ones(len(taus)),  # 闪烁频率噪声项 (常数)
#         np.array(taus) # 随机游走噪声项 (τ)
#     ]).T
#     print(f"设计矩阵: {H}", H.shape)
#     # 计算矩阵条件数
#     cond = np.linalg.cond(H)
#     print(f"设计矩阵条件数: {cond:.2e}")
#     # 构建目标向量 (平方后的测量值)
#     # y 6x1
#     y = adev_measured ** 2
#
#     print(f"目标向量: {y}", y.shape)
#     # 添加非负约束的最小二乘求解
#     theta, _, _, _ = lstsq(H, y)
#     print(f"最小二乘求解参数: {theta}", theta.shape)
#     theta = sqrt(theta)  # 将参数平方根化
#     # 确保所有参数非负
#     theta = np.maximum(theta, 0)
#
#     estimated_params = {
#         'A_wp': theta[0],  # 白相位噪声幅度
#         'A_wf': theta[1],  # 白频率噪声幅度
#         'A_ff': theta[2],  # 闪烁频率噪声幅度
#         'A_rw': theta[3]   # 随机游走噪声幅度
#     }
#
#     return estimated_params


def generate_power_spectrum(taus, adev_measured, num_points=100):
    """
    生成功率谱密度
    :param taus: 积分时间数组 (s)
    :param adev_measured: 测量的Allan标准差数组
    :param num_points: 频率点数
    :return: (功率谱密度数组, 频率数组)
    """
    # 通过Allan方差估计噪声参数h2,h0,h_1,h_2
    _, h_params = estimate_noise_parameters(taus, adev_measured)

    # 生成对数均匀分布的频率点,使用特征频率范围
    f_min = max(1e-10, 1 / (2 * np.pi * taus.max()))  # 最小频率,设置一个较小的正值下限
    f_max = max(1e-9, 1 / (2 * np.pi * taus.min()))  # 最大频率,确保大于f_min
    f = np.logspace(np.log10(f_min), np.log10(f_max), num_points)

    # 计算各噪声分量的功率谱密度
    Sy_components = {
        'wp': h_params['h2'] * f ** 2,  # 白相位噪声:h2*f^2
        'wf': np.full_like(f, h_params['h0']),  # 白频率噪声:h0
        'ff': h_params['h_1'] / f,  # 闪烁频率噪声:h_1/f
        'rw': h_params['h_2'] / f ** 2  # 随机游走噪声:h_2/f^2
    }

    # 叠加各噪声分量得到总功率谱密度
    Sy = np.zeros_like(f)
    for component in Sy_components.values():
        Sy += component

    # 计算时间误差功率谱密度
    Sx = 1 / (4 * np.pi ** 2 * f ** 2) * Sy  # Sx(f) = Sy(f)/(2πf)^2

    return Sy, f, Sx


def plot_allan_comparison(taus, sim_adev, theory_adev, title):
    """
    绘制Allan标准差对比图
    :param taus: 积分时间数组
    :param sim_adev: 仿真得到的Allan标准差
    :param theory_adev: 理论计算的Allan标准差
    :param title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(taus, sim_adev, 'bo-', label='Simulated')
    plt.loglog(taus, theory_adev, 'r--', label='Theoretical')
    plt.xlabel('tau (s)')
    plt.ylabel('Allan Standard Deviation')
    plt.title(title)
    plt.grid(True, which='both')
    plt.legend()
    plt.show()


# ====================================================
# 论文表格4的典型时钟配置示例
# ====================================================
if __name__ == "__main__":
    # 模拟参数
    # USO配置文件
    tau_s = 1  # 采样间隔（秒）
    N = 86400  # 样本数量
    taus_list = [0.1, 1, 10, 100, 1000]  # 积分时间数组，表示分析时钟稳定性的不同时间尺度

    # # 时钟偏差初始值(秒)
    # clock_offsets = 0
    # # 时钟频率偏差(无量纲)
    # clock_freqoffsets = 0# (USO)
    # # 时钟频率线性漂移率(每秒)
    # clock_freqlindrifts = 0
    # # 时钟频率二次漂移率(每秒平方)
    # clock_freqquaddrifts = 0

    # USO Allan方差序列——积分时间数组
    # taus_list = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
    taus = np.array(taus_list)  # 转换为NumPy数组
    measured_adev = np.array([1.6E-13, 7E-14, 7.1E-14, 8E-14, 2.6E-13])

    # USO Allan方差序列——测量值
    # measured_adev = np.array(
    #     [0.0893e-11, 0.0545e-11, 0.0305e-11, 0.0226e-11, 0.0321e-11, 0.0595e-11, 0.1114e-11, 0.1870e-11, 0.2132e-11,
    #      0.1101e-11, 0.1518e-11, 0.1364e-11])
    measured_adev_dict = dict(zip(taus_list, measured_adev))

    # 参数估计（从仿真数据反推噪声参数）
    estimated_params = estimate_noise_parameters(taus, measured_adev)
    # 预测的Allan偏差
    # 计算完整模型的Allan偏差
    Allan_estimated = np.sqrt(
        ((estimated_params[0]) * (1 / taus)) ** 2 +
        ((estimated_params[1]) * (1 / np.sqrt(taus))) ** 2 +
        (estimated_params[2]) ** 2 +
        ((estimated_params[3]) * np.sqrt(taus)) ** 2
    )
    plot_allan_comparison(taus, Allan_estimated, measured_adev, 'Allan标准差对比图')
    # # 分别设置每种噪声为0,研究影响
    # params_wp_zero = estimated_params.copy()
    # params_wp_zero['A_wp'] = 0
    # Allan_wp_zero = np.sqrt(
    #     ((params_wp_zero['A_wf']) * (1 / np.sqrt(taus))) ** 2 +
    #     (params_wp_zero['A_ff']) ** 2 +
    #     ((params_wp_zero['A_rw']) * np.sqrt(taus)) ** 2
    # )
    #
    # params_wf_zero = estimated_params.copy()
    # params_wf_zero['A_wf'] = 0
    # Allan_wf_zero = np.sqrt(
    #     ((params_wf_zero['A_wp']) * (1 / taus)) ** 2 +
    #     (params_wf_zero['A_ff']) ** 2 +
    #     ((params_wf_zero['A_rw']) * np.sqrt(taus)) ** 2
    # )
    #
    # params_ff_zero = estimated_params.copy()
    # params_ff_zero['A_ff'] = 0
    # Allan_ff_zero = np.sqrt(
    #     ((params_ff_zero['A_wp']) * (1 / taus)) ** 2 +
    #     ((params_ff_zero['A_wf']) * (1 / np.sqrt(taus))) ** 2 +
    #     ((params_ff_zero['A_rw']) * np.sqrt(taus)) ** 2
    # )
    #
    # params_rw_zero = estimated_params.copy()
    # params_rw_zero['A_rw'] = 0
    # Allan_rw_zero = np.sqrt(
    #     ((params_rw_zero['A_wp']) * (1 / taus)) ** 2 +
    #     ((params_rw_zero['A_wf']) * (1 / np.sqrt(taus))) ** 2 +
    #     (params_rw_zero['A_ff']) ** 2
    # )
    #
    # # 绘制对比图
    # plt.figure(figsize=(10, 6))
    # plt.loglog(taus, Allan_estimated, 'k-', label='完整模型')
    # plt.loglog(taus, Allan_wp_zero, 'r--', label='无白相位噪声')
    # plt.loglog(taus, Allan_wf_zero, 'g--', label='无白频率噪声')
    # plt.loglog(taus, Allan_ff_zero, 'b--', label='无闪烁频率噪声')
    # plt.loglog(taus, Allan_rw_zero, 'y--', label='无随机游走噪声')
    # plt.loglog(taus, measured_adev, 'ko', label='测量值')
    # plt.xlabel('积分时间 τ (s)')
    # plt.ylabel('Allan标准差 σ(τ)')
    # plt.title('不同噪声分量对Allan标准差的影响')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # estimated_params_new_dict = dict(zip(taus_list, Allan_estimated))

    # # 生成时钟误差
    # generator = ClockErrorGenerator(tau_s=tau_s)
    # y_total, x, _ = generator.generate_clock_errors(estimated_params, N)
    # # 生成时间序列
    # t = np.arange(N) * tau_s
    # clock_offset = clock_offsets + clock_freqoffsets * t + clock_freqlindrifts * t ** 2 / 2 + clock_freqquaddrifts * t ** 3 / 3

    # # 生成各噪声分量
    # _, _, components = generator.generate_clock_errors(estimated_params, N)

11
# # 绘制四种噪声的频率偏差图
# plt.figure(figsize=(10, 10))
# plt.subplot(211)
# plt.plot(t, components['wp'], 'r-', label='白相位噪声', alpha=0.7)
# plt.plot(t, components['wf'], 'g-', label='白频率噪声', alpha=0.7)
# plt.plot(t, components['ff'], 'b-', label='闪烁频率噪声', alpha=0.7)
# plt.plot(t, components['rw'], 'y-', label='随机游走噪声', alpha=0.7)
# plt.xlabel('时间 (s)')
# plt.ylabel('频率偏差 (s/s)')
# plt.title('四种噪声的频率偏差')
# plt.grid(True)
# plt.legend()
#
# # 计算各噪声分量的钟差
# clock_wp = np.cumsum(components['wp']) * tau_s
# clock_wf = np.cumsum(components['wf']) * tau_s
# clock_ff = np.cumsum(components['ff']) * tau_s
# clock_rw = np.cumsum(components['rw']) * tau_s
#
# # 绘制四种噪声的钟差图
# plt.subplot(212)
# plt.plot(t, clock_wp, 'r-', label='白相位噪声', alpha=0.7)
# plt.plot(t, clock_wf, 'g-', label='白频率噪声', alpha=0.7)
# plt.plot(t, clock_ff, 'b-', label='闪烁频率噪声', alpha=0.7)
# plt.plot(t, clock_rw, 'y-', label='随机游走噪声', alpha=0.7)
# plt.xlabel('时间 (s)')
# plt.ylabel('钟差 (s)')
# plt.title('四种噪声的钟差')
# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
# plt.show()


# # 模型一：频率白噪声 + 频率闪烁噪声
# model1_params = {
#     'A_wp': 0,  # 无白相位噪声
#     'A_wf': estimated_params['A_wf'],  # 保留频率白噪声
#     'A_ff': estimated_params['A_ff'],  # 保留频率闪烁噪声
#     'A_rw': 0  # 无随机游走频率噪声
# }
#
# # 模型二：完整噪声模型
# model2_params = {
#     'A_wp': estimated_params['A_wp'],  # 白相位噪声
#     'A_wf': estimated_params['A_wf'],  # 频率白噪声
#     'A_ff': estimated_params['A_ff'],  # 频率闪烁噪声
#     'A_rw': estimated_params['A_rw']   # 随机游走频率噪声
# }
#
# # 生成两种模型的时钟误差
# y_model1, x_model1, _ = generator.generate_clock_errors(model1_params, N)
# y_model2, x_model2, _ = generator.generate_clock_errors(model2_params, N)
#
# # 计算两种模型的完整时钟偏差
# clock_offset_model1 = clock_offsets + clock_freqoffsets * t + clock_freqlindrifts * t**2/2 + clock_freqquaddrifts * t**3/3 + x_model1
# clock_offset_model2 = clock_offsets + clock_freqoffsets * t + clock_freqlindrifts * t**2/2 + clock_freqquaddrifts * t**3/3 + x_model2
#
# # # 绘制三种钟差对比图
# plt.figure(figsize=(12, 6))
# plt.plot(t, clock_offset, 'b-', label='测量值钟差', linewidth=1.5, alpha=0.7)
# plt.plot(t, clock_offset_model1, 'r--', label='模型一钟差(WF+FF)', linewidth=1.5, alpha=0.7)
# plt.plot(t, clock_offset_model2, 'g:', label='模型二钟差(完整模型)', linewidth=1.5, alpha=0.7)
# plt.xlabel('时间 (s)', fontsize=12)
# plt.ylabel('钟差 (s)', fontsize=12)
# plt.title('不同噪声模型的钟差对比', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.show()

# # 计算Allan偏差
# adev = allan_deviation(taus, clock_offset, tau_s)
# #
# # 创建三个子图
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
#
# # Allan偏差图
# ax1.loglog(taus, adev, 'go-', linewidth=1.5, markersize=8)
# ax1.set_xlabel('积分时间 τ (s)', fontsize=12)
# ax1.set_ylabel('Allan偏差 σ(τ) (s/s)', fontsize=12)
# ax1.set_title('Allan偏差随积分时间的变化', fontsize=14)
# ax1.grid(True, which='both', linestyle='--', alpha=0.7)
#
# # 时钟误差图
# ax2.plot(t, clock_offset, 'b-', linewidth=1.5)
# ax2.set_xlabel('时间 (s)', fontsize=12)
# ax2.set_ylabel('钟差 (s)', fontsize=12)
# ax2.set_title('时钟误差序列', fontsize=14)
# ax2.grid(True, linestyle='--', alpha=0.7)
#
# # 频率偏差图
# ax3.plot(t, y_total, 'r-', linewidth=1.5)
# ax3.set_xlabel('时间 (s)', fontsize=12)
# ax3.set_ylabel('频率偏差 (s/s)', fontsize=12)
# ax3.set_title('时钟频率偏差', fontsize=14)
# ax3.grid(True, linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.show()
# # 计算测量值与模拟值之间的差异
# error_model1 = clock_offset - clock_offset_model1
# error_model2 = clock_offset - clock_offset_model2
#
# rmse_model1_cumulative = np.sqrt(np.cumsum(error_model1**2) / np.arange(1, len(error_model1) + 1))
# rmse_model2_cumulative = np.sqrt(np.cumsum(error_model2**2) / np.arange(1, len(error_model2) + 1))
# # #
# # 绘制RMSE对比图
# plt.figure(figsize=(12, 6))
# plt.plot(t, rmse_model1_cumulative, 'b-', label='模型一(WF+FF)', linewidth=1.5, alpha=0.7)
# plt.plot(t, rmse_model2_cumulative, 'r--', label='模型二(完整模型)', linewidth=1.5, alpha=0.7)
# plt.xlabel('时间 (s)', fontsize=12)
# plt.ylabel('累积RMSE (s)', fontsize=12)
# plt.title('不同噪声模型的累积RMSE对比', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.show()
# #
# # 打印RMSE统计信息
# print("\nRMSE统计分析:")
# print("-" * 50)
# print(f"模型一(WF+FF)最终RMSE: {rmse_model1_cumulative[-1]:.2e} 秒")
# print(f"模型二(完整模型)最终RMSE: {rmse_model2_cumulative[-1]:.2e} 秒")
# print(f"模型一平均RMSE: {np.mean(rmse_model1_cumulative):.2e} 秒")
# print(f"模型二平均RMSE: {np.mean(rmse_model2_cumulative):.2e} 秒")
# print(f"模型一最大RMSE: {np.max(rmse_model1_cumulative):.2e} 秒")
# print(f"模型二最大RMSE: {np.max(rmse_model2_cumulative):.2e} 秒")

# # 创建第一个图形 - 钟差序列
# plt.figure(figsize=(12, 6))
# plt.plot(t, clock_offset, 'b-', label='仿真钟差序列', linewidth=1.5)
# plt.xlabel('时间 (s)', fontsize=12)
# plt.ylabel('钟差 (s)', fontsize=12)
# plt.title('时钟误差序列', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.tight_layout()
# plt.show()
#
# # 创建三个独立的图形
# plt.figure(figsize=(10, 6))
# plt.plot(t, clock_offset, 'b-', linewidth=1.5)
# plt.xlabel('时间 (s)', fontsize=12)
# plt.ylabel('钟差 (s)', fontsize=12)
# plt.title('时钟误差累积', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(10, 6))
# plt.plot(t[1:], np.diff(clock_offset), 'r-', linewidth=1.5)
# plt.xlabel('时间 (s)', fontsize=12)
# plt.ylabel('钟差差分 (s/s)', fontsize=12)
# plt.title('时钟误差变化率', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # 绘制频率偏差图
# plt.figure(figsize=(10, 6))
# plt.plot(t, y_total, 'k-', linewidth=1.5)
# plt.xlabel('时间 (s)', fontsize=12)
# plt.ylabel('频率偏差 (s/s)', fontsize=12)
# plt.title('时钟频率偏差', fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# # 计算Allan偏差
# adev = allan_deviation(taus, clock_offset, tau_s)
#
# plt.figure(figsize=(10, 6))
# plt.loglog(taus, adev, 'go-', linewidth=1.5, markersize=8)
# plt.xlabel('积分时间 τ (s)', fontsize=12)
# plt.ylabel('Allan偏差 σ(τ) (s/s)', fontsize=12)
# plt.title('Allan偏差随积分时间的变化', fontsize=14)
# plt.grid(True, which='both', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

#
# # 生成时钟误差
# generator = ClockErrorGenerator(tau_s=tau_s)
# y_total, x, _ = generator.generate_clock_errors(Hydrogen_Maser_params, N)

# 计算Allan方差
# sim_adev = allan_deviation(taus, x, tau_s)


# # 功率谱密度计算
# Sy, f, Sx = generate_power_spectrum(taus, measured_adev, num_points=100)
# plt.figure(figsize=(10, 6))
# # 确保数据为正值
# Sx_positive = np.abs(Sx)  # 取绝对值
# Sx_positive[Sx_positive < 1e-30] = 1e-30  # 设置最小值下限
#
# rad=(Sx_positive * 299792458 /(1064*1E-9))*2 * np.pi
# asd = np.sqrt(Sx_positive)

# # 绘制rad图
# plt.figure(figsize=(10, 6))
# plt.loglog(f, rad, 'bo-', label='仿真结果', linewidth=1.5, markersize=4)
# plt.xlabel('频率 (Hz)', fontsize=12)
# plt.ylabel('相位噪声 (rad/√Hz)', fontsize=12)
# plt.title('USO相位噪声谱密度', fontsize=14)
# plt.grid(True, which='both', linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.xlim(1e-4, 1e0)
# plt.tight_layout()
# plt.show()
#
# # 绘制asd图
# plt.figure(figsize=(10, 6))
# plt.loglog(f, asd, 'ro-', label='仿真结果', linewidth=1.5, markersize=4)
# plt.xlabel('频率 (Hz)', fontsize=12)
# plt.ylabel('幅度谱密度 (s/√Hz)', fontsize=12)
# plt.title('USO时间误差幅度谱密度', fontsize=14)
# plt.grid(True, which='both', linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.xlim(1e-4, 1e0)
# plt.tight_layout()
# plt.show()

# plt.loglog(f, Sx_positive, 'bo-', label='仿真结果', linewidth=1.5, markersize=4)
# plt.xlabel('频率 (Hz)', fontsize=12)
# plt.ylabel('功率谱密度 (s²/Hz)', fontsize=12)
# plt.title('USO时间误差功率谱密度', fontsize=14)
# plt.grid(True, which='both', linestyle='--', alpha=0.7)
# plt.legend(fontsize=10)
# plt.xlim(1e-4, 1e0)  # 设置X轴范围
# plt.tight_layout()
# plt.show()

# # 控制台输出详细误差分析
# print("\n参数估计误差分析:")
# print("{:<8} | {:<12} | {:<12} | {:<10}".format(
#     "τ/s", "实测值", "最小二乘法", "相对误差(%)"))
# print("-" * 50)
# for key in taus_list:
#     true_val = measured_adev_dict[key]
#     est_val = estimated_params_new_dict[key]
#     if true_val == 0:
#         error = 0 if est_val == 0 else np.inf
#     else:
#         error = 100 * abs(est_val - true_val) / true_val
#     print("{:<8} | {:<12.2e} | {:<12.2e} | {:<10.2f}".format(
#         key, true_val, est_val, error))
