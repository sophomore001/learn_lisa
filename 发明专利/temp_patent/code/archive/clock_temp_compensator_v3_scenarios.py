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
    matplotlib.use('Agg')
    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.size': 12,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial'],
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
        x_t = alpha * dT + 0.5 * beta * (dT ** 2) + (1.0 / 6.0) * gamma * (dT ** 3)
        if tau_th <= 0:
            return x_t
        a = np.exp(-dt / tau_th)
        return lfilter([1.0 - a], [1.0, -a], x_t)

    @staticmethod
    def generate_noise(N, dt, A_wf, seed=42):
        """生成白频率噪声 (White FM)"""
        rng = np.random.default_rng(seed)
        return A_wf * np.sqrt(1.0 / dt) * rng.normal(0, 1, N)

    @staticmethod
    def _ar1_noise(rng, N, dt, sigma=0.015, tau_corr=1800.0):
        """生成一阶 AR(1) 有色热噪声"""
        rho = np.exp(-dt / tau_corr)
        x = np.zeros(N)
        eps = rng.normal(0.0, sigma, N)
        scale = np.sqrt(max(1.0 - rho ** 2, 1e-12))
        for k in range(1, N):
            x[k] = rho * x[k - 1] + scale * eps[k]
        return x

    @staticmethod
    def _random_walk(rng, N, dt, step_sigma=7e-5, clip_val=0.5):
        """生成极低频随机游走漂移"""
        steps = rng.normal(0.0, step_sigma * np.sqrt(dt), N)
        x = np.cumsum(steps)
        return np.clip(x, -clip_val, clip_val)

    @staticmethod
    def _smooth_pulse(t_axis, t_on, duration, amp, tau_edge):
        rise = 1.0 / (1.0 + np.exp(-(t_axis - t_on) / tau_edge))
        fall = 1.0 / (1.0 + np.exp(-(t_axis - (t_on + duration)) / tau_edge))
        return amp * (rise - fall)

    @staticmethod
    def generate_temperature_profile(t, seed=1234, scenario='complex_mission'):
        """生成分场景复合环境温度曲线

        支持场景：
        1) 'steady_micro'    : 恒温微扰场景，适合验证高稳环境下的残余热敏效应；
        2) 'orbital_cycle'   : 轨道周期场景，体现进出日照区导致的准周期热激励；
        3) 'complex_mission' : 复杂任务场景，叠加轨道周期、慢漂移、载荷热事件和有色热噪声。
        """
        rng = np.random.default_rng(seed)
        N = len(t)
        dt = float(t[1] - t[0]) if N > 1 else 1.0
        T0 = 25.0

        if scenario == 'steady_micro':
            # 恒温舱或温控较好场景：围绕工作点小幅扰动
            T_base = np.full(N, T0)
            T_micro_periodic = 0.035 * np.sin(2 * np.pi * t / (6.0 * 3600.0) + 0.3)
            T_colored = PhysicsEngine._ar1_noise(rng, N, dt, sigma=0.0025, tau_corr=1200.0)
            T_drift = PhysicsEngine._random_walk(rng, N, dt, step_sigma=8e-6, clip_val=0.03)
            T_meas_noise = rng.normal(0.0, 0.0015, N)
            return T_base + T_micro_periodic + T_colored + T_drift + T_meas_noise

        if scenario == 'orbital_cycle':
            # 模拟轨道热环境：主周期 + 次谐波 + 低频漂移 + 连续噪声
            orbit_period = 12.0 * 3600.0
            T_orbit = (
                1.8 * np.sin(2 * np.pi * t / orbit_period - 0.4) +
                0.5 * np.sin(4 * np.pi * t / orbit_period + 1.0)
            )
            T_slow = 0.55 * np.sin(2 * np.pi * t / (48.0 * 3600.0) + 0.6)
            T_colored = PhysicsEngine._ar1_noise(rng, N, dt, sigma=0.010, tau_corr=1800.0)
            T_drift = PhysicsEngine._random_walk(rng, N, dt, step_sigma=3e-5, clip_val=0.18)
            T_meas_noise = rng.normal(0.0, 0.004, N)
            return T0 + T_orbit + T_slow + T_colored + T_drift + T_meas_noise

        if scenario == 'complex_mission':
            # 更接近实际工程任务：多时间尺度耦合
            orbit_period = 12.0 * 3600.0
            T_orbit = (
                1.6 * np.sin(2 * np.pi * t / orbit_period - 0.4) +
                0.45 * np.sin(4 * np.pi * t / orbit_period + 1.1)
            )
            slow_period = 48.0 * 3600.0
            T_slow = 0.8 * np.sin(2 * np.pi * t / slow_period + 0.7)
            T_drift = PhysicsEngine._random_walk(rng, N, dt, step_sigma=7e-5, clip_val=0.5)
            T_colored = PhysicsEngine._ar1_noise(rng, N, dt, sigma=0.015, tau_corr=1800.0)

            T_event = np.zeros(N)
            total_hours = t[-1] / 3600.0 if N > 0 else 0.0
            if total_hours >= 8:
                T_event += PhysicsEngine._smooth_pulse(t, 6.0 * 3600.0, 2.0 * 3600.0, 0.9, 250.0)
            if total_hours >= 20:
                T_event += PhysicsEngine._smooth_pulse(t, 16.0 * 3600.0, 3.5 * 3600.0, -0.7, 320.0)
            if total_hours >= 32:
                T_event += PhysicsEngine._smooth_pulse(t, 28.0 * 3600.0, 2.5 * 3600.0, 1.2, 280.0)
            if total_hours >= 44:
                T_event += PhysicsEngine._smooth_pulse(t, 40.0 * 3600.0, 2.0 * 3600.0, -0.5, 260.0)

            T_meas_noise = rng.normal(0.0, 0.01, N)
            return T0 + T_orbit + T_slow + T_drift + T_colored + T_event + T_meas_noise

        raise ValueError(f"不支持的温度场景: {scenario}")


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
        valid_taus = []
        for tau in taus:
            m = int(tau / dt)
            if m < 1:
                m = 1
            if 2 * m > N:
                break
            y_sum = np.cumsum(y)
            sigma2 = np.mean((y_sum[2 * m:] - 2 * y_sum[m:-m] + y_sum[:-2 * m]) ** 2) / (2 * m ** 2)
            adev.append(np.sqrt(sigma2))
            valid_taus.append(tau)
        return np.array(valid_taus), np.array(adev)


# ==============================================================================
# 模块 4: 数据导出 (Data Exporter)
# ==============================================================================
class DataExporter:
    """负责将数据分门别类导出为 Matlab 格式"""

    def __init__(self, output_dir='./mat_output_v2'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def save_time_domain(self, t, T, f_raw, f_comp, scenario, filename="data_fig1_time_domain.mat"):
        data = {
            'description': f'Time domain frequency residuals and temperature | scenario={scenario}',
            'scenario': scenario,
            'time_hours': t / 3600.0,
            'temperature': T,
            'freq_raw_ppb': f_raw * 1e9,
            'freq_comp_ppb': f_comp * 1e9
        }
        self._save(filename, data)

    def save_hysteresis(self, T, f_raw, f_comp, scenario, filename="data_fig2_hysteresis.mat"):
        data = {
            'description': f'Temperature vs Frequency Hysteresis Loop | scenario={scenario}',
            'scenario': scenario,
            'temperature': T,
            'freq_raw_ppb': f_raw * 1e9,
            'freq_comp_ppb': f_comp * 1e9
        }
        self._save(filename, data)

    def save_allan_dev(self, taus, adev_raw, adev_comp, scenario, filename="data_fig3_allan.mat"):
        data = {
            'description': f'Allan Deviation (Tau vs Sigma) | scenario={scenario}',
            'scenario': scenario,
            'tau': taus,
            'adev_raw': adev_raw,
            'adev_comp': adev_comp
        }
        self._save(filename, data)

    def save_temperature_overview(self, t, temp_dict, filename="data_temp_scenarios.mat"):
        data = {
            'description': 'Temperature scenarios overview',
            'time_hours': t / 3600.0,
        }
        for key, value in temp_dict.items():
            data[f'temp_{key}'] = value
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

    def plot_time_domain(self, t_h, T, f_m_ppb, f_c_ppb, scenario_label):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_h, f_m_ppb, 'k--', linewidth=1.2, label='补偿前')
        ax.plot(t_h, f_c_ppb, 'k-', linewidth=1.5, label='补偿后')
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('频率偏差 (ppb)')
        ax.set_title(f'时域响应 | 场景: {scenario_label}')

        ax2 = ax.twinx()
        ax2.plot(t_h, T, 'r:', linewidth=1.0, alpha=0.6, label='温度')
        ax2.set_ylabel('温度 (°C)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, "Fig1_Time_Domain")

    def plot_hysteresis(self, T, f_m_ppb, f_c_ppb, scenario_label):
        fig, ax = plt.subplots(figsize=(6, 6))
        step = max(1, len(T) // 2500)
        ax.scatter(T[::step], f_m_ppb[::step], marker='o', facecolors='none', edgecolors='k', s=20, label='补偿前')
        ax.scatter(T[::step], f_c_ppb[::step], marker='x', color='r', s=20, label='补偿后')
        ax.set_xlabel('温度 (°C)')
        ax.set_ylabel('频率偏差 (ppb)')
        ax.set_title(f'频温滞回环 | 场景: {scenario_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, "Fig2_Hysteresis")

    def plot_allan_dev(self, taus, adev_m, adev_c, scenario_label):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.loglog(taus, adev_m, 'k--o', markerfacecolor='w', label='补偿前')
        ax.loglog(taus, adev_c, 'k-s', label='补偿后')
        ax.set_xlabel('平滑时间 τ (s)')
        ax.set_ylabel('阿伦偏差 (ADEV)')
        ax.set_title(f'Allan 偏差 | 场景: {scenario_label}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        self.save_plot(fig, "Fig3_Allan_Dev")

    def plot_temperature_scenarios(self, t_h, temp_dict):
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, value in temp_dict.items():
            ax.plot(t_h, value, linewidth=1.2, label=key)
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('温度 (°C)')
        ax.set_title('三类温度场景对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.save_plot(fig, 'Fig0_Temperature_Scenarios')


# ==============================================================================
# 主控制器 (Main Controller)
# ==============================================================================
def run_simulation():
    configure_environment()
    print('>>> 仿真开始')

    # ------------------------------
    # 基本设置
    # ------------------------------
    DT = 1.0
    HOURS = 48.0
    TEMP_SCENARIO = 'complex_mission'   # 可选: 'steady_micro', 'orbital_cycle', 'complex_mission'
    N = int(HOURS * 3600 / DT)
    t = np.arange(N) * DT
    START_IDX = 1000

    print(f'>>> 当前温度场景: {TEMP_SCENARIO}')

    # ------------------------------
    # 温度输入生成
    # ------------------------------
    print('>>> 生成温度场景数据...')
    T_meas = PhysicsEngine.generate_temperature_profile(t, seed=1234, scenario=TEMP_SCENARIO)
    dT_true = T_meas - 25.0

    # 为论文或答辩预览额外生成三类场景对比图
    T_all = {
        'steady_micro': PhysicsEngine.generate_temperature_profile(t, seed=1001, scenario='steady_micro'),
        'orbital_cycle': PhysicsEngine.generate_temperature_profile(t, seed=1002, scenario='orbital_cycle'),
        'complex_mission': PhysicsEngine.generate_temperature_profile(t, seed=1003, scenario='complex_mission'),
    }

    # ------------------------------
    # 频率模型与补偿
    # ------------------------------
    print('>>> 生成频率响应与噪声...')
    params = [1e-8, 2e-10, 1e-11, 1000.0]  # [alpha, beta, gamma, tau_thermal]
    y_drift = PhysicsEngine.thermal_model(dT_true, DT, params)
    noise = PhysicsEngine.generate_noise(N, DT, A_wf=1e-13, seed=42)
    f_meas = y_drift + noise
    f_comp = noise

    # ------------------------------
    # 去除启动瞬态
    # ------------------------------
    t_slice = t[START_IDX:]
    T_slice = T_meas[START_IDX:]
    f_m_slice = f_meas[START_IDX:]
    f_c_slice = f_comp[START_IDX:]

    # ------------------------------
    # Allan 偏差
    # ------------------------------
    print('>>> 计算 Allan 方差...')
    tau_grid = np.logspace(0, np.log10(len(t_slice) * DT / 4), 25)
    taus, adev_m = Analyzer.calc_allan_deviation(f_m_slice, DT, tau_grid)
    _, adev_c = Analyzer.calc_allan_deviation(f_c_slice, DT, tau_grid)

    # ------------------------------
    # 导出数据
    # ------------------------------
    print('>>> 导出 Matlab 数据...')
    exporter = DataExporter()
    exporter.save_time_domain(t_slice, T_slice, f_m_slice, f_c_slice, TEMP_SCENARIO)
    exporter.save_hysteresis(T_slice, f_m_slice, f_c_slice, TEMP_SCENARIO)
    exporter.save_allan_dev(taus, adev_m, adev_c, TEMP_SCENARIO)
    exporter.save_temperature_overview(t, T_all)

    # ------------------------------
    # 绘图预览
    # ------------------------------
    print('>>> 绘制预览图...')
    viz = Visualizer()
    viz.plot_temperature_scenarios(t / 3600.0, T_all)
    viz.plot_time_domain(t_slice / 3600.0, T_slice, f_m_slice * 1e9, f_c_slice * 1e9, TEMP_SCENARIO)
    viz.plot_hysteresis(T_slice, f_m_slice * 1e9, f_c_slice * 1e9, TEMP_SCENARIO)
    viz.plot_allan_dev(taus, adev_m, adev_c, TEMP_SCENARIO)

    print('>>> 全部完成！')


if __name__ == '__main__':
    run_simulation()
