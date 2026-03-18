# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from scipy.signal import lfilter
import os


# ==============================================================================
# 模块 1: 全局配置 (Config)
# ==============================================================================
def configure_environment():
    """配置绘图后端和字体，解决中文和负号显示问题"""
    matplotlib.use('Agg')  # 非交互模式，适合脚本运行
    plt.rcParams.update({
        'axes.unicode_minus': False,  # 核心：解决负号显示为方块
        'font.size': 12,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial'],  # 优先中文
        'svg.fonttype': 'none',
    })


# ==============================================================================
# 模块 2: 物理引擎 (Physics Engine)
# ==============================================================================
class PhysicsEngine:
    """负责生成所有物理相关的原始数据"""

    @staticmethod
    def thermal_model(dT, dt, params):
        """生成带有热滞后的频率漂移 (Polynomial + Low Pass Filter)"""
        alpha, beta, gamma, tau_th = params

        # 1. 静态温度-频率多项式
        x_t = alpha * dT + 0.5 * beta * (dT ** 2) + (1.0 / 6.0) * gamma * (dT ** 3)

        # 2. 热滞后滤波 (一阶低通)
        if tau_th <= 0:
            return x_t

        a = np.exp(-dt / tau_th)
        # lfilter: y[n] = b0*x[n] + a1*y[n-1] -> y[n] - a*y[n-1] = (1-a)*x[n]
        return lfilter([1.0 - a], [1.0, -a], x_t)

    @staticmethod
    def generate_noise(N, dt, A_wf, seed=42):
        """生成白频率噪声 (White FM)"""
        np.random.seed(seed)
        # White FM 在时域是高斯白噪声，其幅度与 sqrt(1/dt) 成正比
        return A_wf * np.sqrt(1.0 / dt) * np.random.normal(0, 1, N)

    @staticmethod
    def generate_temperature_profile(t):
        """生成环境温度曲线"""
        # 模拟正弦波动 + 随机微扰
        return 17.5 + 27.5 * np.sin(2 * np.pi * t / (3.0 * 3600)) + np.random.normal(0, 0.01, len(t))


# ==============================================================================
# 模块 3: 分析工具 (Analyzer)
# ==============================================================================
class Analyzer:
    """负责数学计算和指标分析"""

    @staticmethod
    def calc_allan_deviation(y, dt, taus):
        """计算阿伦偏差 (Allan Deviation)"""
        adev = []
        N = len(y)
        for tau in taus:
            m = int(tau / dt)
            if 2 * m > N:
                break

            # 使用 cumsum 加速计算重叠 Allan 方差的求和部分
            y_sum = np.cumsum(y)
            # 对应的差分公式
            sigma2 = np.mean((y_sum[2 * m:] - 2 * y_sum[m:-m] + y_sum[:-2 * m]) ** 2) / (2 * m ** 2)
            adev.append(np.sqrt(sigma2))
        return np.array(adev)


# ==============================================================================
# 模块 4: 数据导出 (Data Exporter) - 核心修改部分
# ==============================================================================
class DataExporter:
    """负责将数据分门别类导出为 Matlab 格式"""

    def __init__(self, output_dir='./mat_output_v2'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def save_time_domain(self, t, T, f_raw, f_comp, filename="data_fig1_time_domain.mat"):
        """导出图1：时域残差数据"""
        data = {
            'description': 'Time domain frequency residuals and temperature',
            'time_hours': t / 3600.0,
            'temperature': T,
            'freq_raw_ppb': f_raw * 1e9,  # 转换为 ppb 方便 Matlab 直接画
            'freq_comp_ppb': f_comp * 1e9
        }
        self._save(filename, data)

    def save_hysteresis(self, T, f_raw, f_comp, filename="data_fig2_hysteresis.mat"):
        """导出图2：滞后环数据"""
        data = {
            'description': 'Temperature vs Frequency Hysteresis Loop',
            'temperature': T,
            'freq_raw_ppb': f_raw * 1e9,
            'freq_comp_ppb': f_comp * 1e9
        }
        self._save(filename, data)

    def save_allan_dev(self, taus, adev_raw, adev_comp, filename="data_fig3_allan.mat"):
        """导出图3：Allan方差数据"""
        data = {
            'description': 'Allan Deviation (Tau vs Sigma)',
            'tau': taus,
            'adev_raw': adev_raw,
            'adev_comp': adev_comp
        }
        self._save(filename, data)

    def _save(self, name, data_dict):
        path = os.path.join(self.output_dir, name)
        scipy.io.savemat(path, data_dict)
        print(f"[Export] {name} 已保存至 {self.output_dir}")


# ==============================================================================
# 模块 5: 可视化 (Visualizer)
# ==============================================================================
class Visualizer:
    """负责绘制 matplotlib 图片"""

    def __init__(self, output_dir='./img_output_v2'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def save_plot(self, fig, name):
        fig.savefig(os.path.join(self.output_dir, name + ".svg"), format='svg', bbox_inches='tight')
        fig.savefig(os.path.join(self.output_dir, name + ".png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_time_domain(self, t_h, T, f_m_ppb, f_c_ppb):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_h, f_m_ppb, 'k--', linewidth=1.2, label='补偿前')
        ax.plot(t_h, f_c_ppb, 'k-', linewidth=1.5, label='补偿后')
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('频率偏差 (ppb)')

        ax2 = ax.twinx()
        ax2.plot(t_h, T, 'r:', linewidth=1.0, alpha=0.6, label='温度')
        ax2.set_ylabel('温度 (°C)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, "Fig1_Time_Domain")

    def plot_hysteresis(self, T, f_m_ppb, f_c_ppb):
        fig, ax = plt.subplots(figsize=(6, 6))
        step = 60  # 降采样以免点太密集
        ax.scatter(T[::step], f_m_ppb[::step], marker='o', facecolors='none', edgecolors='k', s=20, label='补偿前')
        ax.scatter(T[::step], f_c_ppb[::step], marker='x', color='r', s=20, label='补偿后')

        ax.set_xlabel('温度 (°C)')
        ax.set_ylabel('频率偏差 (ppb)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, "Fig2_Hysteresis")

    def plot_allan_dev(self, taus, adev_m, adev_c):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.loglog(taus, adev_m, 'k--o', markerfacecolor='w', label='补偿前')
        ax.loglog(taus, adev_c, 'k-s', label='补偿后')

        ax.set_xlabel('平滑时间 τ (s)')
        ax.set_ylabel('阿伦偏差 (ADEV)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        self.save_plot(fig, "Fig3_Allan_Dev")


# ==============================================================================
# 主控制器 (Main Controller)
# ==============================================================================
def run_simulation():
    # 1. 初始化环境
    configure_environment()
    print(">>> 仿真开始")

    # 2. 设置仿真参数
    DT = 1.0
    HOURS = 6.0
    N = int(HOURS * 3600 / DT)
    t = np.arange(N) * DT

    # 忽略启动瞬态的索引 (为了画图好看，导出数据也从这里开始)
    START_IDX = 1000

    # 3. 产生物理数据
    print(">>> 生成物理模型数据...")
    T_meas = PhysicsEngine.generate_temperature_profile(t)
    dT_true = T_meas - 25.0  # 相对25度的温差

    # 参数: [alpha, beta, gamma, tau_thermal]
    params = [1e-8, 2e-10, 1e-11, 1000.0]
    y_drift = PhysicsEngine.thermal_model(dT_true, DT, params)
    noise = PhysicsEngine.generate_noise(N, DT, A_wf=1e-13)

    # 合成测量值与理想补偿值
    f_meas = y_drift + noise
    f_comp = noise  # 理想补偿：假设完全去掉了漂移

    # 4. 数据切片 (去除启动瞬态，用于绘图和导出)
    t_slice = t[START_IDX:]
    T_slice = T_meas[START_IDX:]
    f_m_slice = f_meas[START_IDX:]
    f_c_slice = f_comp[START_IDX:]

    # 5. 计算 Allan 方差
    print(">>> 计算 Allan 方差...")
    taus = np.logspace(0, np.log10(len(t_slice) * DT / 4), 25)
    adev_m = Analyzer.calc_allan_deviation(f_m_slice, DT, taus)
    adev_c = Analyzer.calc_allan_deviation(f_c_slice, DT, taus)

    # 6. 导出数据 (分为三个独立的 .mat 文件)
    print(">>> 导出 Matlab 数据...")
    exporter = DataExporter()

    # 文件 1: 时域
    exporter.save_time_domain(t_slice, T_slice, f_m_slice, f_c_slice)
    # 文件 2: 滞后
    exporter.save_hysteresis(T_slice, f_m_slice, f_c_slice)
    # 文件 3: Allan
    exporter.save_allan_dev(taus, adev_m, adev_c)

    # 7. 绘图 (Python 端预览)
    print(">>> 绘制预览图...")
    viz = Visualizer()
    viz.plot_time_domain(t_slice / 3600, T_slice, f_m_slice * 1e9, f_c_slice * 1e9)
    viz.plot_hysteresis(T_slice, f_m_slice * 1e9, f_c_slice * 1e9)
    viz.plot_allan_dev(taus, adev_m, adev_c)

    print(">>> 全部完成！")


if __name__ == "__main__":
    run_simulation()