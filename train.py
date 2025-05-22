import os
import time
from tqdm import tqdm
import torch
import math
import vessl

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from agent.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

from math import ceil
from torch.utils.data._utils.collate import default_collate

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
            cost, _ = model(move_to(bat, configs.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=configs.eval_batch_size), disable=configs.no_progress_bar)
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
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], configs.run_name))
    step = epoch * (configs.epoch_size // configs.batch_size)
    start_time = time.time()

    if not configs.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    if not configs.no_vessl:
        vessl.log(payload={"learnrate_pg0": optimizer.param_groups[0]['lr']}, step=step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=configs.graph_size, num_samples=configs.epoch_size, case=configs.case))

    dataset_len = len(training_dataset)
    batch_size = configs.batch_size
    num_batches = ceil(dataset_len / batch_size)
    iterator = range(num_batches)

    print('OK?')

    # if not configs.no_progress_bar:
    #     iterator = tqdm(iterator, total=num_batches)

    print('OK??')

    # training_dataloader = DataLoader(training_dataset, batch_size=configs.batch_size, num_workers=1, pin_memory=False, persistent_workers=True)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    print('OK')

    for batch_id in iterator:
        print(step)
        # 1) 인덱스에 따라 샘플 리스트 생성
        start = batch_id * batch_size
        end = min(start + batch_size, dataset_len)
        samples = [training_dataset[i] for i in range(start, end)]
        print('OK?')
        # 2) default_collate로 텐서 배치 생성
        batch = default_collate(samples)
        print('OK??')
        # 3) 실제 학습 함수 호출
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

    # for batch_id, batch in enumerate(tqdm(training_dataloader, disable=configs.no_progress_bar)):
    #     print(step)
    #     train_batch(
    #         model,
    #         optimizer,
    #         baseline,
    #         epoch,
    #         batch_id,
    #         step,
    #         batch,
    #         tb_logger,
    #         configs
    #     )
    #
    #     step += 1
    #     print(step)

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
        vessl.log(payload={"val_avg_reward": avg_reward}, step=step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


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
    print(1)
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, configs.device)
    bl_val = move_to(bl_val, configs.device) if bl_val is not None else None
    print(2)
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)
    print(3)
    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    print(4)
    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    print(5)
    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    print(6)
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, configs.max_grad_norm)
    optimizer.step()
    print(7)
    # Logging
    if step % int(configs.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, configs)