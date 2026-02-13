import torch
import os
import pickle
import numpy as np
import scipy.stats as st

from torch.utils.data import Dataset
from torch.nn.functional import pad
from environment.nesting.state_nesting import StateNESTING
from utils.beam_search import beam_search

from torch_geometric.data import Data, Batch


class NESTING(object):
    NAME = 'nesting'

    @staticmethod
    def build_nearest_neighbor_pi(dataset):
        loc = dataset['loc']  # (B, G, 2), G = 2*M
        start = dataset['start']  # (B, 2)
        batch_size, graph_size, _ = loc.size()
        device = loc.device

        assert graph_size % 2 == 0
        M = graph_size // 2

        # -----------------------------
        # pair 구성 사전 정보
        # -----------------------------
        B, G, _ = loc.size()
        node_ids = torch.arange(G, device=device)
        pair_id_of_node = node_ids // 2  # (G,)

        # 각 pair 내 반대쪽 node 인덱스 사전 계산 (배치 독립적)
        opposite_node = torch.zeros(G, dtype=torch.long, device=device)
        for p in range(M):
            pickup = 2 * p
            delivery = 2 * p + 1
            opposite_node[pickup] = delivery
            opposite_node[delivery] = pickup

        # -----------------------------
        # 최근접 이웃 구성 (전체 노드 대상 → 선택 후 반대쪽으로 이동)
        # -----------------------------
        visited_pair = torch.zeros(B, M, dtype=torch.bool, device=device)
        pi_nn = torch.zeros(B, M, dtype=torch.long, device=device)  # 선택한 entry node

        current_pos = start.clone()  # (B, 2)

        for step in range(M):
            # 현재 위치에서 모든 노드까지 거리 (전체 대상)
            cur_exp = current_pos[:, None, :].expand(B, G, 2)
            dist = (loc - cur_exp).pow(2).sum(-1).sqrt()  # (B, G)

            # 이미 방문한 pair의 노드들만 배제
            visited_for_nodes = visited_pair[:, pair_id_of_node]  # (B, G)
            dist[visited_for_nodes] = float('inf')

            # ★ 전체 노드 중 가장 가까운 node 선택 (entry node)
            next_entry_node = dist.argmin(dim=1)  # (B,)
            pi_nn[:, step] = next_entry_node

            # 해당 pair 방문 표시
            next_pair = pair_id_of_node[next_entry_node]
            visited_pair[torch.arange(B), next_pair] = True

            # ★ 변경: 선택한 node의 반대쪽 node 위치로 이동
            next_exit_node = opposite_node[next_entry_node]  # (B,)
            current_pos = loc[torch.arange(B), next_exit_node]  # (B, 2)

        return pi_nn


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

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, case=1, edge_threshold=0.3):
        super(NESTINGDataset, self).__init__()

        self.edge_threshold = edge_threshold  # GAT용 에지 임계값

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
        # 기존 데이터
        data_dict = self.data[idx]  # {'loc': (2N,2), 'loc_paired': (2N,2), 'start': (2,)}

        # GAT용 PyG Data 생성
        return self._to_gat_data(data_dict)

    def _to_gat_data(self, data_dict, K_percent=1.0):
        loc = data_dict['loc']  # (2N, 2)
        loc_paired = data_dict['loc_paired']  # (2N, 2)
        start = data_dict['start']  # (2,)

        N_total = loc.size(0)  # 2 * nesting_size

        # [loc, loc_paired] concat (4차원)
        x = torch.cat([loc, loc_paired], dim=-1)  # (2N, 4)

        # 2. 에지
        edge_index = []

        with torch.no_grad():
            diff = loc.unsqueeze(1) - loc.unsqueeze(0)  # (N_total, N_total, 2)
            dist_mat = torch.norm(diff, dim=-1)  # (N_total, N_total)

        for i in range(N_total):
            # 자기 자신, 같은 pair는 제외하기 위해 mask
            mask = torch.ones(N_total, dtype=torch.bool, device=loc.device)
            mask[i] = False
            pair_idx = i // 2
            mask[pair_idx * 2: pair_idx * 2 + 2] = False  # 같은 pair 두 개 다 제외

            valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # 후보 j들
            valid_dists = dist_mat[i, valid_indices]

            # K%만큼의 이웃 개수
            k = max(1, int(len(valid_indices) * K_percent))

            # 가장 가까운 k개 이웃 선택
            k_dists, k_idx = torch.topk(valid_dists, k, largest=False)
            neighbors = valid_indices[k_idx]

            for j in neighbors.tolist():
                edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 3. PyG Data
        data = Data(
            x=x,  # (2N, 2) or (2N, 4)
            edge_index=edge_index,  # (2, E)
            start_pos=start.float(),  # (2,)
            num_nodes=N_total  # 메타데이터
        )
        return data