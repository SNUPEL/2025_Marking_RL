import torch
import os
import pickle
import numpy as np
import scipy.stats as st

from torch.utils.data import Dataset
from torch.nn.functional import pad
from environment.nesting.state_nesting import StateNESTING
from utils.beam_search import beam_search


class NESTING(object):
    NAME = 'nesting'

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size, _ = dataset['loc'].size()
        sorted_pi = pi.data.sort(1)[0] // 2
        # Check that sequences are valid, i.e. contain 0 to n -1
        assert (torch.arange(int(graph_size / 2), out=pi.data.new()).view(1, -1).expand(batch_size, int(graph_size / 2))
                 == sorted_pi).all(), "Invalid sequence"

        # Gather dataset in order of tour
        pi = pi + 1
        pi_paired = pi - torch.cos(pi * np.pi)
        pi = pad(pi, (0, 1))
        pi_paired = pad(pi_paired, (1, 0)).to(torch.long)

        loc_with_start = torch.cat((dataset['start'][:, None, :], dataset['loc']), 1)

        d1 = loc_with_start.gather(1, pi[..., None].expand(*pi.size(), loc_with_start.size(-1)))
        d2 = loc_with_start.gather(1, pi_paired[..., None].expand(*pi_paired.size(), loc_with_start.size(-1)))
        cost = (d1 - d2).norm(p=2, dim=2).sum(1)

       # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return NESTINGDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateNESTING.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = NESTING.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class NESTINGDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, case=1):
        super(NESTINGDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.data = []
            if isinstance(data, list):
                for args in data[offset:offset + num_samples]:
                    start, loc, loc_paired, *args = args
                    temp = {'loc': torch.tensor(loc, dtype=torch.float),
                            'loc_paired': torch.tensor(loc_paired, dtype=torch.float),
                            'start': torch.tensor(start, dtype=torch.float)}
                    self.data.append(temp)
            else:
                length_max = torch.max(data['loc']).item()
                data['loc'] = data['loc'] / length_max
                data['loc_paired'] = data['loc_paired'] / length_max
                self.data.append(data)
        else:
            if case == 1:
                loc1 = np.random.uniform(size=(num_samples, size, 2))
                loc2 = loc1 + np.random.uniform(size=(num_samples, size, 2), low=-0.1, high=0.1)
                loc2 = np.random.uniform(size=(num_samples, size, 2))
            elif case == 2:
                loc1 = np.random.uniform(size=(num_samples, size, 2))
                loc2 = np.random.uniform(size=(num_samples, size, 2))
            elif case == 3:
                # plate 스팩 데이터 생성 변수
                mean_estimated = [11616.92823418, 2569.503305]
                cov_estimated = [[19010400.78123422, 1497149.40004837],
                                 [1497149.40004837, 502740.1651592]]
                # 다변량 정규 분포 생성
                mvn = st.multivariate_normal(mean=mean_estimated, cov=cov_estimated)

                # 2. 샘플링 함수 (0보다 작은 값을 폐기하고 다시 샘플링)
                def sample_positive_from_mvn(mvn, size=1):
                    samples = []
                    while len(samples) < size:
                        sample = mvn.rvs()  # 다변량 정규 분포에서 샘플링
                        if np.all(sample > 0):  # 모든 값이 0보다 크면 저장
                            samples.append(sample)
                    return np.array(samples)

                # 3. 피팅된 분포에서 양의 값만 가지는 데이터를 샘플링
                data = sample_positive_from_mvn(mvn, num_samples)
                norm = np.max(data, axis=-1)

                loc1 = np.random.uniform(size=(num_samples, size, 2)) * data[:, np.newaxis, :] / norm[:, np.newaxis, np.newaxis]
                loc2 = np.random.uniform(size=(num_samples, size, 2)) * data[:, np.newaxis, :] / norm[:, np.newaxis, np.newaxis]
            else:
                # plate 스팩 데이터 생성 변수
                mean_estimated = [11616.92823418, 2569.503305]
                cov_estimated = [[19010400.78123422, 1497149.40004837],
                                 [1497149.40004837, 502740.1651592]]
                # 다변량 정규 분포 생성
                mvn = st.multivariate_normal(mean=mean_estimated, cov=cov_estimated)

                # 2. 샘플링 함수 (0보다 작은 값을 폐기하고 다시 샘플링)
                def sample_positive_from_mvn(mvn, size=1):
                    samples = []
                    while len(samples) < size:
                        sample = mvn.rvs()  # 다변량 정규 분포에서 샘플링
                        if np.all(sample > 0):  # 모든 값이 0보다 크면 저장
                            samples.append(sample)
                    return np.array(samples)

                # 3. 피팅된 분포에서 양의 값만 가지는 데이터를 샘플링
                data = sample_positive_from_mvn(mvn)
                length = data[0][0]
                width = data[0][1]

                mean_estimated_dx_dy = [0.00361295, 0.01426838]
                cov_estimated_dx_dy = [[0.01983999, -0.0003388],
                                       [-0.0003388, 0.08855117]]

                mvn_dx_dy = st.multivariate_normal(mean=mean_estimated_dx_dy, cov=cov_estimated_dx_dy)

                loc1 = np.zeros((num_samples, size, 2))
                loc2 = np.zeros((num_samples, size, 2))
                for b in range(num_samples):
                    # num_marks개의 마크 생성 (각 마크는 x, y 좌표로 표현)
                    prob = np.random.rand()
                    if prob < 0.22:
                        scaled_dx, scaled_dy = mvn_dx_dy.rvs()
                    elif prob < 0.70:
                        scaled_dx = 0
                        scaled_dy = st.norm.rvs(loc=-0.004586539791899935, scale=0.38251429704714224)
                    else:
                        scaled_dy = 0
                        scaled_dx = st.norm.rvs(loc=0.0012584737281268073, scale=0.21369093615346613)

                    dx = scaled_dx * length
                    dy = scaled_dy * width

                    if scaled_dx < 0:
                        min_x = -scaled_dx
                        max_x = 1
                    else:
                        min_x = 0
                        max_x = 1 - scaled_dx
                    if scaled_dy < 0:
                        min_y = -scaled_dy
                        max_y = 1
                    else:
                        min_y = 0
                        max_y = 1 - scaled_dy

                    x1 = (np.random.rand(size) * (max_x - min_x) + min_x) * length  # x좌표는 0부터 length 사이의 값
                    y1 = (np.random.rand(size) * (max_y - min_y) + min_y) * width  # y좌표는 0부터 width 사이의 값
                    loc1[b] = np.stack((x1, y1), axis=-1)

                    x2 = x1 + dx
                    y2 = y1 + dy
                    loc2[b] = np.stack((x2, y2), axis=-1)

                    norm = max(loc1[b].max(), loc2[b].max())
                    loc1[b] = loc1[b] / norm
                    loc2[b] = loc2[b] / norm

            start = np.zeros((num_samples, 2))
            loc = np.stack([loc1, loc2], axis=2).reshape((num_samples, size * 2, 2))
            loc_paired = np.stack([loc2, loc1], axis=2).reshape((num_samples, size * 2, 2))

            self.data = []
            for i in range(num_samples):
                data = {'loc': torch.FloatTensor(loc[i]),
                        'loc_paired': torch.FloatTensor(loc_paired[i]),
                        'start': torch.FloatTensor(start[i])}
                self.data.append(data)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]