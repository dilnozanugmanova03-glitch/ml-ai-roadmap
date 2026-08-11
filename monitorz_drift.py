import numpy as np
from scipy import stats

train_data=np.random.normal(loc=50, scale=10, size=1000)
production_data=np.random.normal(loc=50, scale=10, size=1000)
statistic, p_value=stats.ks_2samp(train_data, production_data)

print(f"KS Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("DRIFT ANIQLANDI! Train va production data statistik jihatdan farq qiladi.")
else:
    print("Drift yoq. Data tasqimoti barqaror.")