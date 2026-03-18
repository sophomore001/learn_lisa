# -*- coding: utf-8 -*-
import json
import os
from dataclasses import asdict, dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def configure_environment():
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial"],
            "svg.fonttype": "none",
        }
    )


@dataclass
class ModelParams:
    alpha: float
    beta: float
    gamma: float
    tau_th: float


@dataclass
class WorkflowConfig:
    reference_temperature: float = 25.0
    dt: float = 1.0
    duration_hours: float = 48.0
    training_ratio: float = 0.6
    white_fm_level: float = 1.0e-13
    output_dir: str = ""


def static_temperature_offset(delta_t, params):
    return (
        params.alpha * delta_t
        + 0.5 * params.beta * delta_t**2
        + (1.0 / 6.0) * params.gamma * delta_t**3
    )


def dynamic_temperature_offset(static_offset, dt, tau_th, y0=None):
    if tau_th <= 0:
        return static_offset.copy()

    a = np.exp(-dt / tau_th)
    y = np.zeros_like(static_offset)
    y[0] = static_offset[0] if y0 is None else y0
    for k in range(1, len(static_offset)):
        y[k] = a * y[k - 1] + (1.0 - a) * static_offset[k]
    return y


def forward_model(temperature, params, config, y0=None):
    delta_t = temperature - config.reference_temperature
    x_t = static_temperature_offset(delta_t, params)
    y_t = dynamic_temperature_offset(x_t, config.dt, params.tau_th, y0=y0)
    return x_t, y_t


def online_compensate(temperature, frequency_meas, params, config, state0=None):
    delta_t = temperature - config.reference_temperature
    x_t = static_temperature_offset(delta_t, params)

    if params.tau_th <= 0:
        y_t = x_t
    else:
        a = np.exp(-config.dt / params.tau_th)
        y_t = np.zeros_like(x_t)
        y_t[0] = x_t[0] if state0 is None else state0
        for k in range(1, len(x_t)):
            y_t[k] = a * y_t[k - 1] + (1.0 - a) * x_t[k]

    f_comp = frequency_meas - y_t
    return x_t, y_t, f_comp


def generate_white_fm_noise(size, dt, amplitude, seed=42):
    rng = np.random.default_rng(seed)
    return amplitude * np.sqrt(1.0 / dt) * rng.normal(0.0, 1.0, size)


def _ar1_noise(rng, size, dt, sigma, tau_corr):
    rho = np.exp(-dt / tau_corr)
    seq = np.zeros(size)
    innovation = rng.normal(0.0, sigma, size)
    scale = np.sqrt(max(1.0 - rho**2, 1.0e-12))
    for k in range(1, size):
        seq[k] = rho * seq[k - 1] + scale * innovation[k]
    return seq


def _random_walk(rng, size, dt, step_sigma, clip_value):
    increments = rng.normal(0.0, step_sigma * np.sqrt(dt), size)
    walk = np.cumsum(increments)
    return np.clip(walk, -clip_value, clip_value)


def _smooth_pulse(t_axis, t_on, duration, amplitude, tau_edge):
    rise = 1.0 / (1.0 + np.exp(-(t_axis - t_on) / tau_edge))
    fall = 1.0 / (1.0 + np.exp(-(t_axis - (t_on + duration)) / tau_edge))
    return amplitude * (rise - fall)


def generate_temperature_profile(t_axis, seed=1234):
    rng = np.random.default_rng(seed)
    size = len(t_axis)
    dt = float(t_axis[1] - t_axis[0]) if size > 1 else 1.0
    reference = 25.0

    orbital = 1.6 * np.sin(2.0 * np.pi * t_axis / (12.0 * 3600.0) - 0.4)
    orbital += 0.45 * np.sin(4.0 * np.pi * t_axis / (12.0 * 3600.0) + 1.1)
    slow = 0.8 * np.sin(2.0 * np.pi * t_axis / (48.0 * 3600.0) + 0.7)
    colored = _ar1_noise(rng, size, dt, sigma=0.015, tau_corr=1800.0)
    drift = _random_walk(rng, size, dt, step_sigma=7.0e-5, clip_value=0.5)

    event = np.zeros(size)
    total_hours = t_axis[-1] / 3600.0 if size else 0.0
    if total_hours >= 8.0:
        event += _smooth_pulse(t_axis, 6.0 * 3600.0, 2.0 * 3600.0, 0.9, 250.0)
    if total_hours >= 20.0:
        event += _smooth_pulse(t_axis, 16.0 * 3600.0, 3.5 * 3600.0, -0.7, 320.0)
    if total_hours >= 32.0:
        event += _smooth_pulse(t_axis, 28.0 * 3600.0, 2.5 * 3600.0, 1.2, 280.0)
    if total_hours >= 44.0:
        event += _smooth_pulse(t_axis, 40.0 * 3600.0, 2.0 * 3600.0, -0.5, 260.0)

    measurement_noise = rng.normal(0.0, 0.01, size)
    return reference + orbital + slow + colored + drift + event + measurement_noise


def select_quasi_static_samples(temperature, dt, quantile=0.35, min_points=500):
    slew = np.abs(np.gradient(temperature, dt))
    threshold = np.quantile(slew, quantile)
    mask = slew <= threshold
    if np.count_nonzero(mask) < min_points:
        mask = slew <= np.quantile(slew, 0.55)
    return mask


def estimate_static_initial_params(temperature, frequency_meas, config):
    mask = select_quasi_static_samples(temperature, config.dt)
    delta_t = temperature[mask] - config.reference_temperature
    design = np.column_stack(
        [delta_t, 0.5 * delta_t**2, (1.0 / 6.0) * delta_t**3]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, frequency_meas[mask], rcond=None)
    return coeffs


def estimate_tau_initial(temperature, frequency_meas, config, coeffs):
    static_guess = static_temperature_offset(
        temperature - config.reference_temperature,
        ModelParams(coeffs[0], coeffs[1], coeffs[2], 1.0),
    )

    upper = max(600.0, min(config.duration_hours * 3600.0 / 3.0, 20000.0))
    tau_grid = np.logspace(np.log10(max(5.0, 2.0 * config.dt)), np.log10(upper), 80)

    best_tau = float(tau_grid[0])
    best_rmse = np.inf
    for tau in tau_grid:
        dynamic_guess = dynamic_temperature_offset(static_guess, config.dt, tau)
        rmse = np.sqrt(np.mean((frequency_meas - dynamic_guess) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_tau = float(tau)
    return best_tau


def identify_model_params(temperature, frequency_meas, config):
    coeffs = estimate_static_initial_params(temperature, frequency_meas, config)
    tau_init = estimate_tau_initial(temperature, frequency_meas, config, coeffs)

    def residuals(theta):
        params = ModelParams(theta[0], theta[1], theta[2], np.exp(theta[3]))
        _, y_hat = forward_model(temperature, params, config)
        return y_hat - frequency_meas

    theta0 = np.array([coeffs[0], coeffs[1], coeffs[2], np.log(tau_init)], dtype=float)
    result = least_squares(
        residuals,
        theta0,
        method="trf",
        loss="soft_l1",
        f_scale=max(np.std(frequency_meas), 1.0e-13),
        max_nfev=300,
    )

    identified = ModelParams(
        alpha=float(result.x[0]),
        beta=float(result.x[1]),
        gamma=float(result.x[2]),
        tau_th=float(np.exp(result.x[3])),
    )

    init_params = ModelParams(
        alpha=float(coeffs[0]),
        beta=float(coeffs[1]),
        gamma=float(coeffs[2]),
        tau_th=float(tau_init),
    )
    return init_params, identified, result


def calc_allan_deviation(series, dt, tau_grid):
    valid_taus = []
    adev = []
    cumulative = np.cumsum(series)
    size = len(series)
    for tau in tau_grid:
        m = int(round(tau / dt))
        if m < 1:
            m = 1
        if 2 * m >= size:
            break
        sigma2 = np.mean(
            (cumulative[2 * m:] - 2.0 * cumulative[m:-m] + cumulative[:-2 * m]) ** 2
        ) / (2.0 * m**2)
        valid_taus.append(float(tau))
        adev.append(float(np.sqrt(sigma2)))
    return np.asarray(valid_taus), np.asarray(adev)


def temperature_frequency_correlation(temperature, frequency):
    corr_matrix = np.corrcoef(temperature, frequency)
    return float(corr_matrix[0, 1])


def hysteresis_loop_area(temperature, frequency):
    return float(abs(np.trapz(frequency, temperature)))


def ensure_output_dirs(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)


def save_summary(base_dir, summary):
    summary_path = os.path.join(base_dir, "patent_demo_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    return summary_path


def plot_results(base_dir, time_hours, temperature, f_meas, f_comp, taus, adev_meas, adev_comp):
    fig_dir = os.path.join(base_dir, "figures")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_hours, f_meas * 1.0e9, "k--", label="补偿前")
    ax.plot(time_hours, f_comp * 1.0e9, "k-", label="补偿后")
    ax.set_xlabel("时间 (小时)")
    ax.set_ylabel("频率偏差 (ppb)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(time_hours, temperature, "r:", alpha=0.55, label="温度")
    ax2.set_ylabel("温度 (°C)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    fig.savefig(os.path.join(fig_dir, "validation_time_domain.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    step = max(1, len(temperature) // 2000)
    ax.scatter(
        temperature[::step],
        f_meas[::step] * 1.0e9,
        s=16,
        facecolors="none",
        edgecolors="k",
        label="补偿前",
    )
    ax.scatter(
        temperature[::step],
        f_comp[::step] * 1.0e9,
        s=16,
        marker="x",
        c="r",
        label="补偿后",
    )
    ax.set_xlabel("温度 (°C)")
    ax.set_ylabel("频率偏差 (ppb)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "validation_hysteresis.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(taus, adev_meas, "k--o", markerfacecolor="w", label="补偿前")
    ax.loglog(taus, adev_comp, "k-s", label="补偿后")
    ax.set_xlabel("积分时间 τ (s)")
    ax.set_ylabel("Allan 偏差")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(fig_dir, "validation_allan.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_patent_demo():
    configure_environment()

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "output")
    )
    ensure_output_dirs(base_dir)

    config = WorkflowConfig(output_dir=base_dir)
    total_samples = int(config.duration_hours * 3600.0 / config.dt)
    time_axis = np.arange(total_samples) * config.dt
    temperature = generate_temperature_profile(time_axis, seed=1234)

    true_params = ModelParams(
        alpha=7.0e-11,
        beta=-1.2e-11,
        gamma=1.1e-12,
        tau_th=1450.0,
    )

    _, true_dynamic = forward_model(temperature, true_params, config)
    noise = generate_white_fm_noise(total_samples, config.dt, config.white_fm_level, seed=42)
    frequency_meas = true_dynamic + noise

    split_index = int(total_samples * config.training_ratio)
    train_temperature = temperature[:split_index]
    train_frequency = frequency_meas[:split_index]
    valid_temperature = temperature[split_index:]
    valid_frequency = frequency_meas[split_index:]
    valid_time_hours = time_axis[split_index:] / 3600.0

    init_params, identified_params, solver_result = identify_model_params(
        train_temperature, train_frequency, config
    )

    _, identified_dynamic_train = forward_model(train_temperature, identified_params, config)
    state0 = float(identified_dynamic_train[-1])
    _, estimated_dynamic_valid, compensated_valid = online_compensate(
        valid_temperature,
        valid_frequency,
        identified_params,
        config,
        state0=state0,
    )

    tau_grid = np.logspace(0, np.log10(max(len(valid_frequency) * config.dt / 6.0, 10.0)), 24)
    taus, adev_meas = calc_allan_deviation(valid_frequency, config.dt, tau_grid)
    _, adev_comp = calc_allan_deviation(compensated_valid, config.dt, tau_grid)

    metrics = {
        "rmse_before": float(np.sqrt(np.mean(valid_frequency**2))),
        "rmse_after": float(np.sqrt(np.mean(compensated_valid**2))),
        "corr_before": temperature_frequency_correlation(valid_temperature, valid_frequency),
        "corr_after": temperature_frequency_correlation(valid_temperature, compensated_valid),
        "loop_area_before": hysteresis_loop_area(valid_temperature, valid_frequency),
        "loop_area_after": hysteresis_loop_area(valid_temperature, compensated_valid),
    }
    metrics["rmse_improvement_ratio"] = float(
        metrics["rmse_before"] / max(metrics["rmse_after"], 1.0e-30)
    )
    metrics["corr_reduction"] = float(abs(metrics["corr_before"]) - abs(metrics["corr_after"]))
    metrics["loop_area_reduction_ratio"] = float(
        metrics["loop_area_before"] / max(metrics["loop_area_after"], 1.0e-30)
    )

    plot_results(
        base_dir,
        valid_time_hours,
        valid_temperature,
        valid_frequency,
        compensated_valid,
        taus,
        adev_meas,
        adev_comp,
    )

    summary = {
        "config": asdict(config),
        "true_params": asdict(true_params),
        "initial_params": asdict(init_params),
        "identified_params": asdict(identified_params),
        "solver": {
            "cost": float(solver_result.cost),
            "success": bool(solver_result.success),
            "status": int(solver_result.status),
            "message": str(solver_result.message),
            "nfev": int(solver_result.nfev),
        },
        "metrics": metrics,
    }

    summary_path = save_summary(base_dir, summary)
    print(">>> patent workflow completed")
    print("summary:", summary_path)
    for key, value in summary["identified_params"].items():
        print(f"{key}: {value:.6e}")
    print("rmse improvement:", f"{metrics['rmse_improvement_ratio']:.3f}x")


if __name__ == "__main__":
    run_patent_demo()
