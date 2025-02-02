import numpy as np

def gen_random_tsp(num_cities: int, seed: int):
    rng = np.random.default_rng(seed=seed)

    # 座標
    locations = rng.random(size=(num_cities, 2))

    # 距離行列
    x = locations[:, 0]
    y = locations[:, 1]
    distances = np.sqrt(
        (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
        + (y[:, np.newaxis] - y[np.newaxis, :]) ** 2
    )

    return locations, distances
