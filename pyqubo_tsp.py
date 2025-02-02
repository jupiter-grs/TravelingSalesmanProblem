import numpy as np
from pyqubo import Array, Constraint
from dwave.samplers import SimulatedAnnealingSampler
import common

NUM_CITIES = 16
locations, distances = common.gen_random_tsp(NUM_CITIES, seed=1)

q = Array.create("q", (NUM_CITIES, NUM_CITIES), "BINARY")

objective = sum([distances[i, j] * q[n, i] * q[(n+1) % NUM_CITIES, j]
                 for n in range(NUM_CITIES)
                 for j in range(NUM_CITIES)
                 for i in range(NUM_CITIES)
                 ])

# 各行が one-hot
row_constraints = []
for n in range(NUM_CITIES):
    v = sum(q[n, :]) - 1
    row_constraints.append(Constraint(v * v, f"row-{n}"))

# 各列が one-hot
col_constraints = []
for i in range(NUM_CITIES):
    v = sum(q[:, i]) - 1
    col_constraints.append(Constraint(v * v, f"col-{n}"))

constraints = row_constraints + col_constraints


M = np.amax(distances)  # 制約条件の強さを設定
H = objective + M * sum(constraints)

model = H.compile()

# 実行
sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample(model.to_bqm(), num_reads=1000)

# 確認
decoded_answer = model.decode_sampleset(sampleset)
feasibles = [dec_sol for dec_sol in decoded_answer if len(dec_sol.constraints(only_broken=True)) == 0]
if len(feasibles) == 0:
    raise RuntimeError("At least one of the constraints is not satisfied.")

best = min(feasibles, key=lambda v: v.energy)

route = []
for n in range(NUM_CITIES):
    for i in range(NUM_CITIES):
        if best.array("q", (n, i)) > 0.5:
            route.append(i)
route.append(route[0])

print(route)
print(best.energy)
