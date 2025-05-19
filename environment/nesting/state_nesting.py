import torch
import numpy as np

from typing import NamedTuple

from scipy.spatial import distance_matrix

from utils.boolmask import mask_long2bool, mask_long_scatter

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class StateNESTING(NamedTuple):
    # Fixed input
    coords: torch.Tensor
    distance: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    distance_mask: torch.Tensor
    distance_min: torch.Tensor
    distance_max: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        start = input['start']
        loc = input['loc']

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        # visited_ = torch.zeros(batch_size, 1, n_loc + 1, dtype=torch.uint8, device=loc.device)
        # visited_ = visited_.scatter(-1, torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)[:, :, None], 1)

        distance = (input['loc'].unsqueeze(2) - input['loc'].unsqueeze(1)).norm(p=2, dim=-1)
        temp = ~torch.block_diag(*[torch.ones((2, 2)) for _ in range(int(distance.size(1) / 2))]).type(torch.bool)
        distance_mask = temp.expand_as(distance).to(loc.device)
        # distance = distance[mask.expand_as(distance)].reshape(distance.size(0), distance.size(1), distance.size(2) - 2)

        return StateNESTING(
            coords=torch.cat((start[:, None, :], loc), -2),
            distance=distance,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['start'][:, None, :],
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            distance_mask=distance_mask,
            distance_min = torch.min(distance, dim=-1)[0].unsqueeze(-1),
            distance_max = torch.max(distance, dim=-1)[0].unsqueeze(-1),
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        prev_a_paired = (prev_a + torch.cos(prev_a * np.pi)).to(torch.long)

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            visited_ = visited_.scatter(-1, prev_a_paired[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)
            visited_ = mask_long_scatter(visited_, prev_a_paired)

        cur_coord = self.coords[self.ids, prev_a_paired]

        # distance_mask = self.distance_mask.scatter_(-1, prev_a.unsqueeze(-1).expand(-1, self.distance_mask.size(1), -1), 0)
        # distance_mask = distance_mask.scatter_(-1, prev_a_paired.unsqueeze(-1).expand(-1, distance_mask.size(1), -1), 0)
        #
        # distance_masked = torch.where(distance_mask, self.distance, torch.tensor(float("inf"), device=self.distance.device))
        # distance_min = torch.min(distance_masked, dim=-1)[0].unsqueeze(-1)
        # distance_min = torch.where(~torch.isinf(distance_min), distance_min, 0.0)
        #
        # distance_masked = torch.where(distance_mask, self.distance,torch.tensor(float("-inf"), device=self.distance.device))
        # distance_max = torch.max(distance_masked, dim=-1)[0].unsqueeze(-1)
        # distance_max = torch.where(~torch.isinf(distance_max), distance_max, 0.0)
        #
        # return self._replace(prev_a=prev_a, visited_=visited_,
        #                      lengths=lengths, cur_coord=cur_coord, i=self.i + 1, distance_mask=distance_mask,
        #                      distance_min=distance_min, distance_max=distance_max)

        return self._replace(prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= int((self.coords.size(-2) - 1) / 2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        mask1 = self.visited > 0
        # mask2 = torch.BoolTensor(
        #     np.linalg.norm(self.coords[:, 1:, :].cpu().numpy() - self.cur_coord.cpu().numpy(), 2, -1) > 0.4
        # ).unsqueeze(1).to(mask1.device)
        # mask = mask1 | mask2
        # rollback = torch.all(mask, dim=-1)
        # mask[rollback] = mask1[rollback]
        return mask1

    def construct_solutions(self, actions):
        return actions