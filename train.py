import os
import time
from tqdm import tqdm
import torch
import math
import csv

from torch_geometric.loader import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F

from agent.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, configs):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            if isinstance(bat, dict):
                bat = bat['data'] if 'data' in bat else bat

            if hasattr(bat, 'x'):  # PyG Data
                bat = bat.to(configs.device)
                cost, _ = model(bat)
            else:  # 기존 dict
                bat = move_to(bat, configs.device)
                cost, _ = model(bat)
            return cost.data.cpu()

    pyg_loader = DataLoader(dataset, batch_size=configs.eval_batch_size, follow_batch=['x'], num_workers=0, pin_memory=True)

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(pyg_loader, disable=configs.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, configs):
    csv_filename = f'training_log.csv'
    csv_path = os.path.join(configs.save_dir, csv_filename)

    # 디렉토리 자동 생성 (없으면 만듦!)
    os.makedirs(configs.save_dir, exist_ok=True)

    # 첫 epoch에서만 헤더 생성
    if epoch == configs.epoch_start:
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'step', 'batch_id', 'avg_cost', 'val_reward',
                    'lr', 'reinforce_loss', 'grad_norm', 'epoch_time'
                ])
            print(f"CSV logging started: {csv_path}")
        except Exception as e:
            print(f"CSV header failed: {e}")

    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], configs.run_name))
    step = epoch * (configs.epoch_size // configs.batch_size)
    start_time = time.time()

    if not configs.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    if not configs.no_vessl:
        import vessl
        vessl.log(payload={"learnrate_pg0": optimizer.param_groups[0]['lr']}, step=step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=configs.graph_size, num_samples=configs.epoch_size, case=configs.case))
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        follow_batch=['x'],
        num_workers=0,
        pin_memory=True
    )

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=configs.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            configs
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (configs.checkpoint_epochs != 0 and epoch % configs.checkpoint_epochs == 0) or epoch == configs.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(configs.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, configs)

    if not configs.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)
    if not configs.no_vessl:
        import vessl
        vessl.log(payload={"val_avg_reward": avg_reward}, step=step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    # Epoch 결과 저장
    try:
        epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, step, -1, 0, avg_reward.item(),
                optimizer.param_groups[0]['lr'], 0, 0, epoch_time
            ])
    except Exception as e:
        print(f"⚠️ CSV append failed: {e}")


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        configs
):
    if hasattr(batch, 'x'):  # PyG Data
        x = batch.to(configs.device)
        bl_val = None  # baseline은 별도 처리
    else:  # 기존 dict batch
        x, bl_val = baseline.unwrap_batch(batch)
        x = move_to(x, configs.device)
        bl_val = move_to(bl_val, configs.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, log_p = model(x, return_log_p=True)

    print(f"Batch nodes: {x.size(0)}, edge max: {batch.edge_index.max() if 'batch' in locals() else 'N/A'}")
    if hasattr(x, 'validate'):  # x가 Data/Batch
        x.validate(raise_on_error=True)

    bl_val, bl_loss = baseline.eval(x, cost)  # (B,), scalar

    adv = (cost - bl_val).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    adv = adv / 0.5
    pg_loss = (adv * log_likelihood).mean()

    probs = log_p.exp()
    entropy_per_step = -(probs * log_p).sum(dim=-1)  # (batch, seq_len)
    entropy = entropy_per_step.mean()  # scalar

    if epoch < 50:
        lambda_entropy = 2e-3
    elif epoch < 100:
        lambda_entropy = 1e-3
    else:
        lambda_entropy = 0.0

    loss = pg_loss - lambda_entropy * entropy

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    bl_loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, configs.max_grad_norm)
    optimizer.step()

    if torch.isnan(cost).any() or torch.isnan(log_likelihood).any():
        print("NaN detected in cost or log_likelihood")

    # Logging
    if step % int(configs.log_step) == 0:
        # cost_nn, _ = baseline.warmup_baseline.eval(x, cost)
        # avg_cost_nn = cost_nn.mean().item()
        avg_cost_nn = 0
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, loss, bl_loss, avg_cost_nn, tb_logger, configs)