import torch
import torch.nn as nn
import torch.nn.functional as F

import utils 
import numpy as np

class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        # 네트워크 구조 빌드
        self.build()

    def build(self):
        # 상태 임베딩
        self.state_em = nn.Embedding(2, 2)  # 상태에 대한 임베딩 (2개의 상태를 2차원으로 임베딩)
        self.process_coords = nn.Linear(4, 16)  # 4차원 입력을 16차원으로 변환하는 선형 레이어
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)  # 1D 합성곱 레이어

    def forward(self, traj, config):
        # 'lngs'와 'lats'는 경도와 위도 값으로, 차원을 추가합니다.
        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)

        # 'states'는 상태를 임베딩합니다.
        states = self.state_em(traj['states'].long())

        # 경도, 위도, 상태 임베딩을 하나로 병합
        locs = torch.cat((lngs, lats, states), dim=2)

        # 좌표를 16차원 벡터로 변환
        locs = torch.tanh(self.process_coords(locs))  # F.tanh 대신 torch.tanh 사용
        locs = locs.permute(0, 2, 1)  # 차원을 변경하여 (batch_size, 16, sequence_length)로 변환

        # 1D 합성곱 적용 후 활성화 함수 ELU 적용
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)  # 다시 차원을 (batch_size, sequence_length, num_filter)로 변환

        # 로컬 경로에 대한 거리 계산
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'], config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim=2)  # 거리에 대해 차원을 추가하여 (batch_size, sequence_length, 1)

        # 합성곱 결과와 거리를 병합
        conv_locs = torch.cat((conv_locs, local_dist), dim=2)

        return conv_locs
