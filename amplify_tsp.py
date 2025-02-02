import os
from datetime import timedelta
import numpy as np
from amplify import VariableGenerator, einsum, Poly, one_hot, FixstarsClient, solve
import common

NUM_CITIES = 16
locations, distances = common.gen_random_tsp(NUM_CITIES, seed=1)

gen = VariableGenerator()
q = gen.array("Binary", shape=(NUM_CITIES + 1, NUM_CITIES))
q[NUM_CITIES, :] = q[0, :]


objective: Poly = einsum("ij,ni,nj->", distances, q[:-1], q[1:])


# 最後の行を除いた q の各行のうち一つのみが 1 である制約
row_constraints = one_hot(q[:-1], axis=1)

# 最後の行を除いた q の各列のうち一つのみが 1 である制約
col_constraints = one_hot(q[:-1], axis=0)

constraints = row_constraints + col_constraints


constraints *= np.amax(distances)  # 制約条件の強さを設定
model = objective + constraints


client = FixstarsClient()
client.token = os.getenv("AMPLIFY")
client.parameters.timeout = timedelta(milliseconds=1000)  # タイムアウト 1000 ミリ秒

# ソルバーの設定と結果の取得
result = solve(model, client)
if len(result) == 0:
    raise RuntimeError("At least one of the constraints is not satisfied.")

best = result.best
# print(best)

route = []
for i, j in zip(*np.where(q.evaluate(best.values) > 0.5)):
    route.append(int(j))

print(route)
print(best.objective)
