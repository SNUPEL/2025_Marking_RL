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


class PPORolloutBuffer:
    def __init__(self):
        self.costs = []
        self.logp_olds = []
        self.values = []
        self.actions = []
        self.batches = []
        self.inputs = []

    def add_batch(self, cost, logp_old, value, actions, x):
        # 모두 (B, ...) 텐서라고 가정
        self.costs.append(cost.detach())
        self.logp_olds.append(logp_old.detach())
        self.values.append(value.detach())
        self.actions.append(actions.detach())
        self.inputs.append(x.detach().cpu())

    def get_all(self):
        # 나중에 PPO 업데이트 전에 한 번에 concat
        costs = torch.cat(self.costs, dim=0)       # (N,)
        logp_olds = torch.cat(self.logp_olds, 0)   # (N, T) 또는 (N,)
        values = torch.cat(self.values, 0)         # (N,)
        actions = torch.cat(self.actions, 0)       # (N, T)
        inputs = torch.cat(self.inputs, 0)  # (N, input_dim ...)
        return costs, logp_olds, values, actions, inputs

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

    buffer=PPORolloutBuffer()

    rollout_costs = []
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=configs.no_progress_bar)):
        batch_cost = collect_batch(
            model,
            baseline,
            batch,
            buffer,
            configs
        )

        if not configs.no_tensorboard and batch_id % configs.log_step == 0:
            tb_logger.log_value('train/rollout_batch_cost', batch_cost, step)
        rollout_costs.append(batch_cost)

        step += 1

    avg_rollout_cost = sum(rollout_costs) / len(rollout_costs)

    # 1) 버퍼에서 전부 꺼내기
    all_costs, all_logp_olds, all_values, all_actions, all_x = buffer.get_all()  # (N,...)

    # 2) advantage / returns 전체 기준으로 한 번 계산
    advantages = (all_costs - all_values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = all_costs.detach()

    # ---------- 2) PPO 업데이트 ----------
    K_epochs = configs.ppo_epochs
    eps_clip = configs.ppo_clip
    vf_coef = configs.vf_coef
    ent_coef = configs.ent_coef

    N = all_costs.size(0)
    mb_size = configs.minibatch_size

    total_pg_loss, total_v_loss, total_entropy = 0.0, 0.0, 0.0
    total_actor_grad_norm, total_critic_grad_norm = 0.0, 0.0
    num_updates = 0

    for _ in range(K_epochs):
        perm = torch.randperm(N)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]

            cost_b = all_costs[idx].to(configs.device)
            logp_old_b = all_logp_olds[idx].to(configs.device)
            value_b = all_values[idx].to(configs.device)
            act_b = all_actions[idx].to(configs.device)
            adv_b = advantages[idx].to(configs.device)
            returns_b = returns[idx].to(configs.device)
            x_b = all_x[idx].to(configs.device)

            # 새 policy 통과
            cost_new, _, log_p_new = model(x_b, return_log_p=True, return_pi=False)
            v_new = baseline.critic(x_b)

            if log_p_new.dim() == 3:
                logp_new_b = log_p_new.gather(2, act_b.unsqueeze(-1)).squeeze(-1)
            else:
                logp_new_b = log_p_new

            if logp_new_b.dim() == 2:
                logp_new_red = logp_new_b.mean(dim=1)
                logp_old_red = logp_old_b.mean(dim=1)
            else:
                logp_new_red = logp_new_b
                logp_old_red = logp_old_b

            ratio = torch.exp(logp_new_red - logp_old_red)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv_b
            pg_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(v_new, returns_b)
            probs = log_p_new.exp()
            entropy_per_step = -(probs * log_p_new).sum(-1)
            entropy = entropy_per_step.mean()

            loss = pg_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norms, grad_norms_clipped = clip_grad_norms(
                optimizer.param_groups, configs.max_grad_norm
            )
            optimizer.step()

            actor_grad_norm, critic_grad_norm = grad_norms_clipped
            total_pg_loss += pg_loss.item()
            total_v_loss += value_loss.item()
            total_entropy += entropy.item()
            total_actor_grad_norm += actor_grad_norm
            total_critic_grad_norm += critic_grad_norm
            num_updates += 1

    avg_pg_loss = total_pg_loss / num_updates
    avg_v_loss = total_v_loss / num_updates
    avg_entropy = total_entropy / num_updates
    avg_actor_grad = total_actor_grad_norm / num_updates
    avg_critic_grad = total_critic_grad_norm / num_updates

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
        tb_logger.log_value('train/rollout_avg_cost', avg_rollout_cost, epoch)
        tb_logger.log_value('train/ppo_policy_loss', avg_pg_loss, epoch)
        tb_logger.log_value('train/ppo_value_loss', avg_v_loss, epoch)
        tb_logger.log_value('train/entropy', avg_entropy, epoch)
        tb_logger.log_value('train/actor_grad_norm', avg_actor_grad, epoch)
        tb_logger.log_value('train/critic_grad_norm', avg_critic_grad, epoch)
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

def collect_batch(model, baseline, batch, buffer, configs):
    # 1) 입력 준비 (네 train_batch와 동일)
    if hasattr(batch, 'x'):
        x = batch.to(configs.device)
        # 필요한 텐서만 추출
        x_tensor = batch.x
    else:
        x, _ = baseline.unwrap_batch(batch)
        x = move_to(x, configs.device)
        x_tensor = batch['x']

        # 2) 정책에서 샘플 + log_p, critic value 계산
    #    (GATModel forward를 약간 수정해서 pi와 per-step log_p를 같이 돌려받는다고 가정)
    #    cost: (B,), _log_p: (B, T, A), pi: (B, T)
    cost, log_likelihood, _log_p, pi = model(x, return_log_p=True, return_pi=True)

    # critic baseline
    v = baseline.critic(x)          # (B,)

    # 3) 선택된 action에 대한 log_prob_old 계산
    logp_old = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)   # (B, T)

    # 4) 버퍼에 저장
    buffer.add_batch(cost, logp_old, v, pi, x_tensor)

    batch_cost = cost.mean().item()
    return batch_cost