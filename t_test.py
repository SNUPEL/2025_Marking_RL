import numpy as np
from scipy import stats

def extract_stats_equal_var(list1, list2, alpha=0.05):
    """
    두 독립 표본(등분산 가정)에 대한
    MS, SE(diff), t, df, p 값을 계산해 딕셔너리로 반환.
    """
    n1, n2 = len(list1), len(list2)
    mean1, mean2 = np.mean(list1), np.mean(list2)
    var1, var2 = np.var(list1, ddof=1), np.var(list2, ddof=1)

    # 1) MS (pooled variance)
    ms_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    # 2) SE(diff)
    se_diff = np.sqrt(ms_pooled * (1/n1 + 1/n2))

    # 3) t 통계량
    t_stat = (mean1 - mean2) / se_diff

    # 4) 자유도
    df = n1 + n2 - 2

    # 5) p-값(양측)
    p_value = 2 * stats.t.sf(np.abs(t_stat), df)

    return {
        "MS(pooled)": ms_pooled,
        "SE(diff)": se_diff,
        "t": t_stat,
        "df": df,
        "p": p_value,
    }


# 두 집단 데이터
group_A = [3.13, 3.93, 3.13, 3.5, 4.37, 4, 4, 3.37, 2.67, 4, 3.9, 4.27, 4.6, 4.17, 4.57, 4.4, 3.5]
group_B = [3.1, 3, 3.2, 3.17, 3.07, 4.6, 4, 3, 2.57, 4, 3.83, 3.8, 2.9, 4.13, 4.57, 3, 3.17]

results = extract_stats_equal_var(group_A, group_B)
for k, v in results.items():
    print(f"{k:12}: {v: .4f}" if isinstance(v, float) else f"{k:12}: {v}")