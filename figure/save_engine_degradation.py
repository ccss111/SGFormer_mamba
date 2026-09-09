import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.preprocessing import CmapssDataSet, gen_sequence, load_dataset


def _resolve_model_path(model_path, trials_dir, sub_dataset):
    if model_path:
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(f"指定模型不存在: {model_path}")

    candidates = [
        os.path.join(trials_dir, f"model_{sub_dataset}.pkl"),
        *glob.glob(os.path.join(trials_dir, f"model_{sub_dataset}_*.pkl")),
        *glob.glob(os.path.join(trials_dir, f"best_model_{sub_dataset}_*.pkl")),
    ]
    candidates = [path for path in candidates if os.path.exists(path)]
    if candidates:
        return max(candidates, key=os.path.getmtime)
    raise FileNotFoundError(
        f"在 {trials_dir} 下找不到 {sub_dataset} 对应模型，请使用 --model-path 显式指定。"
    )


def _enable_dropout_only(module):
    if isinstance(module, nn.Dropout):
        module.train()


def _mc_dropout_predict(model, x_batch, mc_samples, device, use_amp=False):
    model.eval()
    model.apply(_enable_dropout_only)
    predictions = []
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else None
    )
    with torch.no_grad():
        for _ in range(max(1, int(mc_samples))):
            if autocast_ctx is None:
                prediction, _ = model(x_batch)
            else:
                with autocast_ctx:
                    prediction, _ = model(x_batch)
            predictions.append(prediction.detach().float())
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)


def _normal_interval(mean_values, variance_values, interval_alpha):
    confidence = np.clip(float(interval_alpha), 0.0, 100.0)
    tail_probability = confidence / 200.0
    if tail_probability == 0.0:
        return mean_values.copy(), mean_values.copy()

    standard_deviation = np.sqrt(np.maximum(variance_values, 0.0))
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    lower_z = normal.icdf(torch.tensor(tail_probability)).item()
    upper_z = normal.icdf(torch.tensor(1.0 - tail_probability)).item()
    return (
        mean_values + lower_z * standard_deviation,
        mean_values + upper_z * standard_deviation,
    )


def _make_engine_dataset(group, engine_id, seq_length, source, max_rul, y_test=None):
    if engine_id not in group.groups:
        raise ValueError(f"{source} 集不存在发动机 {engine_id}，可选范围为 1-{len(group)}")

    engine = group.get_group(engine_id)
    if source == "train":
        features = engine.iloc[:, 2:-1]
        labels = engine.iloc[:, -1].to_numpy(dtype=np.float32)
    else:
        features = engine.iloc[:, 2:]
        observed_cycles = len(features)
        hidden_rul = float(y_test.iloc[engine_id - 1, 0]) * max_rul
        labels = np.arange(
            observed_cycles - 1 + hidden_rul,
            hidden_rul - 1,
            -1,
            dtype=np.float32,
        )
        labels = np.clip(labels, 0.0, max_rul) / max_rul

    if len(features) < seq_length:
        raise ValueError(
            f"发动机 {engine_id} 只有 {len(features)} 个周期，少于 sequence-len={seq_length}"
        )
    windows = np.asarray(list(gen_sequence(features, seq_length)), dtype=np.float32)
    window_labels = labels[seq_length - 1:]
    dataset = CmapssDataSet(windows, window_labels.reshape(-1, 1))
    cycles = np.arange(seq_length, len(features) + 1, dtype=np.int32)
    true_rul = window_labels * max_rul
    return DataLoader(dataset, batch_size=32, shuffle=False), cycles, true_rul


def _collect_predictions(model, loader, max_rul, device, mc_samples, eval_batch_size, use_amp):
    prediction_loader = DataLoader(loader.dataset, batch_size=eval_batch_size, shuffle=False)
    means = []
    variances = []
    for x_batch, _ in prediction_loader:
        mean, variance = _mc_dropout_predict(
            model, x_batch.to(device), mc_samples, device, use_amp
        )
        means.append(mean.cpu().reshape(-1))
        variances.append(variance.cpu().reshape(-1))
    mean_rul = torch.cat(means).numpy() * max_rul
    variance_rul = torch.cat(variances).numpy() * (max_rul ** 2)
    return mean_rul, variance_rul


def _interval_metrics(true_rul, pred_lower, pred_upper, target_picp=0.95,
                      cwc_eta=50.0, cwc_gamma=1.0):
    true_rul = np.asarray(true_rul, dtype=np.float64).reshape(-1)
    pred_lower = np.asarray(pred_lower, dtype=np.float64).reshape(-1)
    pred_upper = np.asarray(pred_upper, dtype=np.float64).reshape(-1)
    if not (len(true_rul) == len(pred_lower) == len(pred_upper)):
        raise ValueError("true values and interval bounds must have the same length")
    if len(true_rul) == 0:
        raise ValueError("cannot calculate interval metrics for an empty engine trajectory")

    lower = np.minimum(pred_lower, pred_upper)
    upper = np.maximum(pred_lower, pred_upper)
    picp = float(np.mean((true_rul >= lower) & (true_rul <= upper)))
    data_range = float(np.max(true_rul) - np.min(true_rul))
    pinaw = float(np.mean(upper - lower) / data_range) if data_range > 0 else 0.0
    target_picp = float(np.clip(target_picp, 0.0, 1.0))
    if picp >= target_picp:
        cwc = pinaw
    else:
        cwc = pinaw * (1.0 + cwc_gamma * np.exp(-cwc_eta * (picp - target_picp)))
    return {"PICP": picp, "PINAW": pinaw, "CWC": float(cwc)}


def _save_metrics_csv(path, rows):
    import csv

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model_path", "model_code", "sub_dataset", "engine_source",
                        "engine_id", "PICP", "PINAW", "CWC"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_degradation_plot(path, sub_dataset, engine_id, source, cycles, true_rul,
                           pred_rul, pred_lower, pred_upper):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_cycles = cycles - cycles[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.fill_between(plot_cycles, pred_lower, pred_upper, color="#2a66f0", alpha=0.15,
                    label="95% prediction interval")
    ax.plot(plot_cycles, true_rul, color="#35a853", linewidth=1.7, label="True RUL")
    ax.plot(plot_cycles, pred_rul, color="#2a66f0", linewidth=1.5, label="Predicted RUL")
    ax.plot(plot_cycles, pred_lower, color="#ff5a52", linewidth=0.9, alpha=0.8,
            label="Lower PI")
    ax.plot(plot_cycles, pred_upper, color="#174ea6", linewidth=0.9, alpha=0.8,
            label="Upper PI")
    ax.set_xlabel("Forecast cycles")
    ax.set_ylabel("Remaining useful life (cycles)")
    ax.set_title(f"{sub_dataset} {source} engine {engine_id} degradation curve")
    ax.set_xlim(0, plot_cycles[-1])
    ax.set_ylim(bottom=0, top=max(1.0, float(max(np.max(true_rul), np.max(pred_upper))) * 1.08))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.tick_params(which="minor", length=2.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="绘制单台发动机的 RUL 退化曲线")
    parser.add_argument("--sub-dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--engine-source", choices=["train", "test"], default="train",
                        help="默认使用训练集完整寿命轨迹；test 使用测试集隐藏终点 RUL")
    parser.add_argument("--engine-id", type=int, default=1)
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--smooth-rate", type=int, default=30)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--dataset-root", default=os.path.join(PROJECT_ROOT, "CMAPSSData"))
    parser.add_argument("--model-path", nargs="+", default=None,
                        help="一个或多个训练好的模型路径")
    parser.add_argument("--model-code", nargs="+", default=None,
                        help="与模型路径按位置对应的模型代号")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument("--interval-alpha", type=float, default=5.0)
    parser.add_argument("--target-picp", type=float, default=0.95,
                        help="CWC目标覆盖率，默认0.95")
    parser.add_argument("--cwc-eta", type=float, default=50.0,
                        help="CWC覆盖率惩罚强度")
    parser.add_argument("--cwc-gamma", type=float, default=1.0,
                        help="CWC覆盖率惩罚系数")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--degradation-svg-path", default="")
    parser.add_argument("--metrics-csv", default="",
                        help="指标汇总CSV路径，默认保存到退化曲线目录")
    args = parser.parse_args()

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    model_paths = args.model_path or [_resolve_model_path(
        "", os.path.join(PROJECT_ROOT, "trials"), args.sub_dataset
    )]
    if args.model_code is None:
        model_codes = [os.path.splitext(os.path.basename(path))[0] for path in model_paths]
    elif len(args.model_code) != len(model_paths):
        parser.error("--model-code 的数量必须与 --model-path 的数量相同")
    else:
        model_codes = args.model_code
    model_codes = [os.path.basename(code) for code in model_codes]

    group_train, group_test, y_test = load_dataset(
        args.dataset_root + os.sep, args.sub_dataset, args.max_rul, args.sequence_len,
        True, args.smooth_rate
    )
    group = group_train if args.engine_source == "train" else group_test
    loader, cycles, true_rul = _make_engine_dataset(
        group, args.engine_id, args.sequence_len, args.engine_source, args.max_rul, y_test
    )
    metric_rows = []
    output_dir = os.path.join(
        PROJECT_ROOT,
        "figure",
        "save_engine_degradation",
        f"{args.sub_dataset}-engine{args.engine_id}",
    )
    os.makedirs(output_dir, exist_ok=True)
    for model_path, model_code in zip(model_paths, model_codes):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"指定模型不存在: {model_path}")
        model = torch.load(model_path, map_location=device)
        model.to(device)
        pred_rul, variance_rul = _collect_predictions(
            model, loader, args.max_rul, device, args.mc_samples,
            max(1, args.eval_batch_size), args.amp
        )
        pred_lower, pred_upper = _normal_interval(pred_rul, variance_rul, args.interval_alpha)
        pred_lower = np.clip(pred_lower, 0.0, args.max_rul)
        pred_upper = np.clip(pred_upper, 0.0, args.max_rul)
        metrics = _interval_metrics(
            true_rul, pred_lower, pred_upper, args.target_picp,
            args.cwc_eta, args.cwc_gamma
        )
        metric_rows.append({
            "model_path": model_path,
            "model_code": model_code,
            "sub_dataset": args.sub_dataset,
            "engine_source": args.engine_source,
            "engine_id": args.engine_id,
            **metrics,
        })

        if args.degradation_svg_path and len(model_paths) == 1:
            output_path = os.path.join(output_dir, os.path.basename(args.degradation_svg_path))
        else:
            output_path = os.path.join(
                output_dir,
                f"{args.sub_dataset}_{args.engine_source}_engine{args.engine_id}_"
                f"{model_code}_degradation.svg",
            )
        _save_degradation_plot(
            output_path, args.sub_dataset, args.engine_id, args.engine_source,
            cycles, true_rul, pred_rul, pred_lower, pred_upper
        )
        print(f"model_path:{model_path}")
        print(f"degradation_svg:{output_path}")
        print("interval_metrics: PICP={PICP:.6f}, PINAW={PINAW:.6f}, CWC={CWC:.6f}".format(**metrics))

    metrics_filename = os.path.basename(args.metrics_csv) if args.metrics_csv else (
        f"{args.sub_dataset}_{args.engine_source}_engine{args.engine_id}_interval_metrics.csv"
    )
    metrics_path = os.path.join(output_dir, metrics_filename)
    _save_metrics_csv(metrics_path, metric_rows)
    print(f"metrics_csv:{metrics_path}")


if __name__ == "__main__":
    main()
