import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Pickle 파일 경로 (필요에 따라 변경하세요)
file_path = './data/nesting/nesting80_case4_seed1234.pkl'

# 2. Pickle 로드
with open(file_path, 'rb') as f:
    data = pickle.load(f)
# data: list of tuples (start, loc, loc_paired)

# 3. 샘플 인덱스 지정 (예: 첫 번째 샘플)
sample_idx = 0
start, loc, loc_paired = data[sample_idx]

# 4. numpy 배열로 변환
loc = np.array(loc)           # shape: (nesting_size*2, 2)
loc_paired = np.array(loc_paired)

# 5. 산점도로 시각화
plt.figure()
plt.scatter(loc[:, 0], loc[:, 1], label='loc')
plt.title(f'Sample {sample_idx}: Loc')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')  # 축 비율 동일하게
plt.show()
