import os
import json
import vessl
import pprint as pp

import torch
import torch.optim as optim
# from tensorboard_logger import Logger as TbLogger

from agent.critic_network import CriticNetwork
from configurations import get_configurations
from train import train_epoch, validate, get_inner_model
from baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from agent.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem


def run(configs):
    if not configs.no_vessl:
        vessl.run(organization="snu-eng-dgx-heavy", project="Marking", hp=configs)

    # Pretty print the run args
    pp.pprint(vars(configs))

    # Set the random seed
    torch.manual_seed(configs.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not configs.no_tensorboard:
        tb_logger = TbLogger(os.path.join(configs.log_dir, "{}".format(configs.graph_size), configs.run_name))

    os.makedirs(configs.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(configs.save_dir, "args.json"), 'w') as f:
        json.dump(vars(configs), f, indent=True)

    # Set the device
    configs.device = torch.device("cuda" if configs.use_cuda else "cpu")

    print(torch.cuda.get_device_name(0))

    # Figure out what's the problem
    problem = load_problem(configs.problem)

    # Load data from load_path
    load_data = {}
    assert configs.load_path is None or configs.resume is None, "Only one of load path and resume can be given"
    load_path = configs.load_path if configs.load_path is not None else configs.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model = AttentionModel(
        configs.embedding_dim,
        configs.hidden_dim,
        problem,
        n_encode_layers=configs.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=configs.normalization,
        tanh_clipping=configs.tanh_clipping,
        checkpoint_encoder=configs.checkpoint_encoder,
        shrink_size=configs.shrink_size
    ).to(configs.device)

    if configs.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if configs.baseline == 'exponential':
        baseline = ExponentialBaseline(configs.exp_beta)
    elif configs.baseline == 'critic':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetwork(
                    2,
                    configs.embedding_dim,
                    configs.hidden_dim,
                    configs.n_encode_layers,
                    configs.normalization
                )
            ).to(configs.device)
        )
    elif configs.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, configs)
    else:
        assert configs.baseline is None, "Unknown baseline: {}".format(configs.baseline)
        baseline = NoBaseline()

    if configs.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, configs.bl_warmup_epochs, warmup_exp_beta=configs.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': configs.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': configs.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(configs.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: configs.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(size=configs.graph_size, num_samples=configs.val_size,
                                       filename=configs.val_dataset, case=configs.case)

    if configs.resume:
        epoch_resume = int(os.path.splitext(os.path.split(configs.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if configs.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        configs.epoch_start = epoch_resume + 1

    if configs.eval_only:
        validate(model, val_dataset, configs)
    else:
        for epoch in range(configs.epoch_start, configs.epoch_start + configs.n_epochs):
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


if __name__ == "__main__":
    run(get_configurations())