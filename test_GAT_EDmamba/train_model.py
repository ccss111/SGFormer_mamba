import datetime
import argparse
import os
import sys
import subprocess
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from test_GAT_EDmamba import *
from test_GAT_EDmamba.AsymmetricLoss import AdaptiveAsymmetricMSELoss
from utils import *
# 自动创建 logs 文件夹（不存在就新建）
default_log_dir = os.path.join(PROJECT_ROOT, "logs")
if not os.path.exists(default_log_dir):
    os.makedirs(default_log_dir)


DEFAULT_SEEDS = [2]#, 17, 27, 30, 33, 51, 62, 80, 88, 97


def _build_log_name(model_code, lr, embed_dim, topk, time_tag):
    model_code = str(model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty.")
    lr_str = format(float(lr), "g")
    return (
        f"{model_code}_学习率+{lr_str}_dmodel+{embed_dim}_"
        f"dstate+{topk}_{time_tag}.log"
    )



if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')
    parent_dir = PROJECT_ROOT
    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=os.path.join(parent_dir, 'CMAPSSData') + '/', 
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--sub-dataset', type=str, default='FD002', help='FD001/2/3/4')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr-scheduler', type=str, default='step', choices=['step'],
                        help='learning rate scheduler type (step only)')
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--use-exponential-smoothing', default=True)  
    parser.add_argument('--smooth-rate', type=int, default=40)
    parser.add_argument('--use-spatial-gat', action='store_true', default=True, help='Whether to use spatial GAT')
    parser.add_argument('--gat-hidden-dim', type=int, default=8, help='Hidden size of GAT blocks')
    parser.add_argument('--mamba-d-model', type=int, default=10, help='input/output size of encoder/decoder Mamba blocks')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--mamba-num-layers', type=int, default=2, help='Number of Mamba layers')
    parser.add_argument('--mamba-d-state', type=int, default=16, help='Hidden size of Mamba layers')
    parser.add_argument('--save-model', dest='save_model', action='store_true', default=True,
                        help='save trained models')
    parser.add_argument('--no-save-model', dest='save_model', action='store_false',
                        help='do not save trained models')
    parser.add_argument('--model-code', type=str, default='A',
                        help='模型代号，用于日志命名')
    parser.add_argument('--seed-list', type=int, nargs='+', default=None,
                        help='Override training seeds, e.g. --seed-list 62 80 88 97')
    parser.add_argument('--start-seed', type=int, default=None,
                        help='Start from this seed within the active seed list')
    parser.add_argument('--asym-alpha', type=float, default=2.0,
                        help='Penalty strength for overestimated RUL predictions')
    parser.add_argument('--asym-gamma', type=float, default=5.0,
                        help='Weight boost for low-RUL focus samples')
    parser.add_argument('--asym-delta', type=float, default=0.9,
                        help='Weight multiplier for healthy high-RUL cap samples')
    parser.add_argument('--focus-threshold', type=float, default=35.0,
                        help='RUL threshold below which samples receive focus weighting')
    parser.add_argument('--cap-threshold', type=float, default=125.0,
                        help='RUL threshold above which healthy samples are down-weighted')
    args = parser.parse_args()

    model_code = str(args.model_code).strip()
    if not model_code:
        raise ValueError("--model-code cannot be empty.")

    ablation_preset_info = None
    encoder_hidden_size = args.mamba_d_model
    mamba_num_layers = args.mamba_num_layers
    mamba_d_state = args.mamba_d_state

    run_output_root = os.path.join(default_log_dir, f"{args.sub_dataset}_{model_code}")
    os.makedirs(run_output_root, exist_ok=True)

    seed_sequence = list(DEFAULT_SEEDS if args.seed_list is None else args.seed_list)
    if not seed_sequence:
        raise ValueError("seed sequence is empty. Use --seed-list with at least one seed.")
    if args.start_seed is not None:
        if args.start_seed not in seed_sequence:
            raise ValueError(f"--start-seed {args.start_seed} is not in active seed list: {seed_sequence}")
        seed_sequence = seed_sequence[seed_sequence.index(args.start_seed):]
    print(f"[Seed Plan] {seed_sequence}")

    for num in seed_sequence:

        torch.manual_seed(num)


        train_loader, valid_loader, test_loader, test_loader_last, \
            num_test_windows, train_visualize, engine_id = get_dataloader(
                dir_path=args.dataset_root,
                sub_dataset=args.sub_dataset,
                max_rul=args.max_rul,
                seq_length=args.sequence_len,
                batch_size=args.batch_size,
                use_exponential_smoothing=args.use_exponential_smoothing,
                smooth_rate=args.smooth_rate)

        encoder_input_size = args.feature_num
        encoder = Seq2SeqEncoder(
            input_size=encoder_input_size,
            num_layers=mamba_num_layers,
            num_hidden=encoder_hidden_size,
            d_state=mamba_d_state,
        )
        decoder = Seq2SeqDecoder(
            input_size=encoder_hidden_size,
            num_layers=mamba_num_layers,
            num_hidden=encoder_hidden_size,
            d_state=mamba_d_state,
        )
        model = EncoderDecoder(encoder=encoder, decoder=decoder, use_spatial_gat=args.use_spatial_gat, gat_hidden_dim=args.gat_hidden_dim, mamba_num_layers=2, mamba_d_state=mamba_d_state)
        

        model_type = type(model).__name__

        criterion_train = AdaptiveAsymmetricMSELoss(
            alpha=args.asym_alpha,
            gamma=args.asym_gamma,
            delta=args.asym_delta,
            focus_threshold=args.focus_threshold,
            cap_threshold=args.cap_threshold,
            max_rul=args.max_rul,
        )
        criterion_eval = RMSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )

        seed_output_dir = os.path.join(run_output_root, f"seed{num}")
        os.makedirs(seed_output_dir, exist_ok=True)

        time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_tag = f"{time}_seed{num}"
        log_file_name = _build_log_name(
            model_code=model_code,
            lr=args.lr,
            embed_dim=encoder_hidden_size,
            topk=mamba_d_state,
            time_tag=run_tag,
        )
        log_path = os.path.join(seed_output_dir, log_file_name)

        with open(log_path, "a", encoding="utf-8") as f:

            f.write("-----"+ args.sub_dataset + "-----\n")
            f.write("关键参数:\n")
            f.write(f"序列长度: {args.sequence_len}\n")
            f.write(f"批大小(batch_size): {args.batch_size}\n")
            f.write(f"最大训练轮数(max_epochs): {args.max_epochs}\n")
            f.write(f"随机数种子: {num}\n")
            f.write(f"模型代号(model_code): {args.model_code}\n")
            f.write("模型结构(model_structure): mamba_mamba\n")
            f.write(f"学习率: {args.lr}\n")
            f.write("学习率调度器: step\n")
            f.write(f"Step步长(step_size): {args.step_size}\n")
            f.write(f"Step衰减系数(gamma): {args.gamma}\n")
            f.write(f"use_gat: {args.use_spatial_gat}\n")
            f.write(f"gat_hidden_dim: {args.gat_hidden_dim}\n")
            f.write("编码器: 2层Mamba\n")
            f.write("解码器: 2层Mamba\n")
            f.write(f"encoder_hidden_size: {encoder_hidden_size}\n")
            f.write(f"mamba_num_layers: {mamba_num_layers}\n")
            f.write(f"mamba_d_state: {mamba_d_state}\n")
            f.write("训练损失: AdaptiveAsymmetricMSELoss\n")
            f.write(f"asym_alpha: {args.asym_alpha}\n")
            f.write(f"asym_gamma: {args.asym_gamma}\n")
            f.write(f"asym_delta: {args.asym_delta}\n")
            f.write(f"focus_threshold: {args.focus_threshold}\n")
            f.write(f"cap_threshold: {args.cap_threshold}\n")
            f.write("------------------------------\n")


        train(
            model, train_loader, valid_loader,
            test_loader, args.max_epochs, optimizer,
            scheduler, criterion_train, criterion_eval,
            lines_list=[], patience=args.patience, max_rul=args.max_rul, num_test_windows=num_test_windows,
            device=torch.device('cuda') if not args.no_cuda else torch.device('cpu'), time=time, log_path=log_path,
            checkpoint_dir=seed_output_dir,
            checkpoint_prefix='model_' + args.sub_dataset + '_' + run_tag)

        model_output_dir = seed_output_dir
        os.makedirs(model_output_dir, exist_ok=True)

        saved_model_path = os.path.join(model_output_dir, 'model_' + args.sub_dataset + '_' + run_tag + '.pkl')

        if args.save_model:
            torch.save(model, saved_model_path)
        else:
            # train() 会在验证最优时写入 best_*.pkl
            saved_model_path = os.path.join(model_output_dir, 'best_model_' + args.sub_dataset + '_' + run_tag + '.pkl')

        auto_test_cmd = [
            sys.executable,
            os.path.join(parent_dir, 'test_GAT_EDmamba', 'test_model.py'),
            '--sub-dataset', args.sub_dataset,
            '--smooth-rate', str(args.smooth_rate),
            '--sequence-len', str(args.sequence_len),
            '--feature-num', str(args.feature_num),
            '--dataset-root', args.dataset_root,
            '--max-rul', str(args.max_rul),
            '--batch-size', str(args.batch_size),
            '--model-path', saved_model_path,
        ]
        if args.no_cuda:
            auto_test_cmd.append('--no-cuda')

        test_result = subprocess.run(
            auto_test_cmd,
            cwd=parent_dir,
            capture_output=True,
            text=True,
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("[Auto Test] command: " + " ".join(auto_test_cmd) + "\n")
            if test_result.stdout:
                f.write(test_result.stdout.strip() + "\n")
            if test_result.returncode != 0:
                f.write("[Auto Test][ERROR] return_code={}\n".format(test_result.returncode))
                if test_result.stderr:
                    f.write(test_result.stderr.strip() + "\n")

        if test_result.returncode == 0:
            print("[Auto Test] completed. Result appended to log file.")
            if test_result.stdout:
                print(test_result.stdout.strip())
        else:
            print("[Auto Test][ERROR] Failed to run test_model.py. See log for details.")
