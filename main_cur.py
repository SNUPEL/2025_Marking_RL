import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import pprint as pp

import torch
import torch.optim as optim

from agent.critic_network import CriticNetwork
from configurations import get_configurations
from train import train_epoch, validate, get_inner_model
from baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from agent.attention_model import GATModel
from utils import torch_load_cpu, load_problem, eval_nn_heuristic_on_val


def build_model_and_baseline(configs, load_data):
    problem = load_problem(configs.problem)

    model = GATModel(
        configs.embedding_dim,
        configs.hidden_dim,
        problem,
        n_encode_layers=configs.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=configs.normalization,
        n_heads=configs.n_heads,
        tanh_clipping=configs.tanh_clipping,
        checkpoint_encoder=configs.checkpoint_encoder,
        shrink_size=configs.shrink_size
    ).to(configs.device)

    if configs.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # baseline
    if configs.baseline == 'exponential':
        baseline = ExponentialBaseline(configs.exp_beta)
    elif configs.baseline == 'critic':
        baseline = CriticBaseline(
            CriticNetwork(
                4,
                configs.embedding_dim,
                configs.hidden_dim,
                configs.n_encode_layers,
                configs.n_heads
            ).to(configs.device)
        )
    elif configs.baseline == 'rollout':
        problem = load_problem(configs.problem)
        baseline = RolloutBaseline(model, problem, configs)
    else:
        assert configs.baseline is None, "Unknown baseline: {}".format(configs.baseline)
        baseline = NoBaseline()

    if configs.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, configs.bl_warmup_epochs, model, problem, configs.device)

    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    return model, baseline


def run_single_size(configs, graph_size_idx, graph_size, model, baseline, optimizer, lr_scheduler, tb_logger, load_data):
    # 문제/데이터셋 세팅
    problem = load_problem(configs.problem)
    configs.graph_size = graph_size

    # 스테이지별 초기 LR 설정
    stage_lr = configs.lr_per_stage[graph_size_idx] if configs.lr_per_stage else configs.lr_model
    for param_group in optimizer.param_groups:
        param_group['lr'] = stage_lr

    print(f"Stage {graph_size_idx + 1}/N={graph_size}: LR={stage_lr:.2e}, decay={configs.lr_decay}")

    val_dataset = problem.make_dataset(
        size=configs.graph_size,
        num_samples=configs.val_size,
        filename=configs.val_dataset,
        case=configs.case
    )

    # resume 처리 (있다면)
    if configs.resume:
        epoch_resume = int(os.path.splitext(os.path.split(configs.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if configs.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        configs.epoch_start = epoch_resume + 1

    # NN heuristic 평가
    val_avg_cost_nn = eval_nn_heuristic_on_val(problem, val_dataset, configs.device)
    print(f'Nearest-neighbor heuristic (N={graph_size}) val_avg_cost_nn: {val_avg_cost_nn.item():.4f}')

    # eval-only 모드는 그대로
    if configs.eval_only:
        validate(model, val_dataset, configs)
        return

    # 이 스테이지에서 몇 epoch 돌릴지 결정 (예: 총 n_epochs를 3등분)
    # 필요하면 configs에 따로 n_epochs_20, n_epochs_40 이런 식으로 넣어도 됨
    n_stage_epochs = configs.n_epochs_per_stage if hasattr(configs, "n_epochs_per_stage") \
                      else max(1, configs.n_epochs // 3)

    for local_epoch in range(n_stage_epochs):
        epoch = configs.epoch_start + local_epoch
        train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_dataset,
            problem,
            tb_logger,
            configs
        )

        import gc
        torch.cuda.empty_cache()
        gc.collect()

    # 다음 스테이지에서 이어서 학습하도록 epoch_start 업데이트
    configs.epoch_start += n_stage_epochs


def run_curriculum(configs):
    if not configs.no_vessl:
        import vessl
        from vessl.internal.vessl_run import VesslRun
        VesslRun()

    pp.pprint(vars(configs))

    torch.manual_seed(configs.seed)

    tb_logger = None
    if not configs.no_tensorboard:
        from tensorboard_logger import Logger as TbLogger
        tb_logger = TbLogger(os.path.join(configs.log_dir, "{}".format("curriculum"), configs.run_name))

    os.makedirs(configs.save_dir, exist_ok=True)
    with open(os.path.join(configs.save_dir, "args.json"), 'w') as f:
        json.dump(vars(configs), f, indent=True)

    configs.device = torch.device("cuda" if configs.use_cuda else "cpu")
    if configs.use_cuda:
        print(torch.cuda.get_device_name(0))

    # load_path / resume 처리
    load_data = {}
    assert configs.load_path is None or configs.resume is None, "Only one of load path and resume can be given"
    load_path = configs.load_path if configs.load_path is not None else configs.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # 공통 model, baseline, optimizer, scheduler 생성
    model, baseline = build_model_and_baseline(configs, load_data)

    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': configs.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': configs.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    if 'optimizer' in load_data and configs.resume:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(configs.device)

    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: configs.lr_decay ** epoch
    )

    # 커리큘럼 순서
    curriculum_sizes = getattr(configs, "curriculum_sizes", [20, 40, 80])

    # 전체 epoch_start 초기화
    if not hasattr(configs, "epoch_start"):
        configs.epoch_start = 0

    for stage_idx, gsz in enumerate(curriculum_sizes):
        run_single_size(
            configs,
            graph_size_idx=stage_idx,  # lr_per_stage 인덱스용
            graph_size=gsz,
            model=model,
            baseline=baseline,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tb_logger=tb_logger,
            load_data=load_data,
        )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    configs = get_configurations()
    run_curriculum(configs)
