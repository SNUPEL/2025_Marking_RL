import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from train import rollout, get_inner_model


class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs, model, problem, device):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = NNBaseline(model, problem, device)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)

        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, configs, epoch=0):
        super(Baseline, self).__init__()

        self.problem = problem
        self.configs = configs

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.configs.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.configs.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                size=self.configs.graph_size, num_samples=self.configs.val_size, case=self.configs.case)
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.configs).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.model, dataset, self.configs).view(-1, 1))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.configs).cpu().numpy()

        candidate_mean = candidate_vals.mean()

        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            # Calc p value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.configs.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])

class NNBaseline(Baseline):
    """
    Baseline that uses a fixed NN heuristic policy's cost as baseline.
    - model: actor 모델 (GATModel) – 여기서 problem, embedder, set_decode_type 등을 가져다 씀
    - problem: NESTING / TSP 등 문제 클래스 (build_nearest_neighbor_pi, get_costs 필요)
    - alpha: critic / exp baseline 등과 convex combination 할 때 가중치는
             train_batch 쪽에서 섞는 편이 더 유연하므로,
             이 클래스는 "NN baseline 값만" 돌려주는 역할만 한다고 보는 게 깔끔함.
    """

    def __init__(self, model, problem, device):
        super(NNBaseline, self).__init__()
        self.problem = problem
        self.device = device

        # actor 구조를 그대로 복사해 둘 수도 있고, 그냥 참조만 할 수도 있음.
        # 여기서는 "정책은 항상 최신 모델"을 쓰고, NN 경로만 heuristic으로 쓰기 때문에
        # 굳이 deepcopy는 하지 않고 model 참조만 사용.
        self.model = model

    def eval(self, x, c):
        """
        x : PyG Batch 또는 dict batch (train_batch에서 그대로 넘겨주는 입력)
        c : 실제 cost (B,) – 여기서는 사용 안 함, 인터페이스 맞추기용
        return:
          v : (B,) baseline 값 (NN policy cost)
          l : 0 (baseline 학습 없음)
        """

        # PyG Batch만 고려 (지금 구조 기준)
        if not hasattr(x, 'x'):
            # dict batch라면, train_batch에서 이미 PyG로 변환한 뒤 넣어주는 게 깔끔
            raise ValueError("NNBaseline expects a PyG Batch with attribute 'x'.")

        batch = x.to(self.device)
        B = batch.num_graphs
        N = batch.num_nodes // B

        # actor 쪽과 동일한 방식으로 input_dict 구성
        loc_dummy = batch.x.view(B, N, 4)  # (B, N, 4)
        input_dict = {
            'loc': loc_dummy[:, :, :2],
            'loc_paired': loc_dummy[:, :, 2:],
        }
        if hasattr(batch, 'start_pos'):
            input_dict['start'] = batch.start_pos.view(B, -1)
        else:
            input_dict['start'] = loc_dummy[:, 0, :2]

        with torch.no_grad():
            # 1) NN policy 시퀀스 생성
            pi_nn = self.problem.build_nearest_neighbor_pi(input_dict)  # (B, T)

            # 2) 그 시퀀스의 cost 계산
            cost_nn = self.problem.get_costs(input_dict, pi_nn)[0]     # (B,)

        # Detach: actor 쪽으로 gradient 안 흘리기
        return cost_nn.detach(), 0.0

    def get_learnable_parameters(self):
        # NN baseline은 학습하지 않음 (pure heuristic)
        return []

    def state_dict(self):
        # heuristic baseline이므로 굳이 저장할 건 없음
        return {}

    def load_state_dict(self, state_dict):
        # 로딩할 파라미터 없음
        pass

class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)


