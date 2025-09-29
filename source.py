import random
import numpy as np

def simule_remb(n, params):
    alpha = params["alpha"]
    x0 = params["x0"]
    eta = params["eta"]
    if alpha <= 0 or x0 <= 0 or eta <= 3/2:
        raise ValueError("Les paramètres doivent respecter : alpha > 0, x0 > 0, eta > 3/2.")
    a = 2 * eta - 1
    M = (x0 * alpha**(-eta)) / a
    batch_size = min(max(10000, n), 100000)
    target_acceptance_rate = 0.1
    results = np.zeros(n)
    accepted_count = 0
    max_iterations = 100
    iteration = 0
    while accepted_count < n and iteration < max_iterations:
        U = np.random.uniform(0, 1, size=batch_size)
        try:
            Y = x0 * (1 - U)**(-1 / a)
            mask = (Y > 0)
            Y = Y[mask]
            if len(Y) > 0:
                f_val = (alpha + (Y - x0)**2)**(-eta)
                g_val = a * (x0**a) / (Y**(a + 1))
                mask_valid = (g_val > 0)
                Y = Y[mask_valid]
                f_val = f_val[mask_valid]
                g_val = g_val[mask_valid]
                if len(Y) > 0:
                    ratio = f_val / (M * g_val)
                    U2 = np.random.uniform(0, 1, size=len(Y))
                    accepted_mask = (U2 <= ratio)
                    accepted_values = Y[accepted_mask]
                    remaining = n - accepted_count
                    adding = min(remaining, len(accepted_values))
                    results[accepted_count:accepted_count+adding] = accepted_values[:adding]
                    accepted_count += adding
                    acceptance_rate = len(accepted_values) / len(Y)
                    if acceptance_rate < target_acceptance_rate / 2:
                        batch_size = min(int(batch_size * 1.5), 500000)
                    elif acceptance_rate > target_acceptance_rate * 2:
                        batch_size = max(int(batch_size * 0.8), n - accepted_count)
        except (RuntimeWarning, OverflowError, ValueError):
            batch_size = max(int(batch_size * 0.5), 1000)
        iteration += 1
    if accepted_count < n:
        raise RuntimeError(f"Impossible de générer {n} échantillons valides après {max_iterations} itérations.")
    return results

def simule_meteo(n_days, ptrans):
    pSN, pNS, pNP, pPN = ptrans
    chain = np.zeros(n_days, dtype=np.int8)
    P = np.array([
        [1 - pSN, pSN, 0],
        [pNS, 1 - pNS - pNP, pNP],
        [0, pPN, 1 - pPN]
    ])
    for i in range(1, n_days):
        chain[i] = np.random.choice(3, p=P[chain[i-1]])
    state_map = {0: "S", 1: "N", 2: "P"}
    return [state_map[state] for state in chain]

def msa(n_simul, params):
    s = params["s"]
    N = params["N"]
    pacc = params["pacc"]
    ptrans = params["ptrans"]
    values = []
    P = np.array([
        [1 - ptrans[0], ptrans[0], 0],
        [ptrans[1], 1 - ptrans[1] - ptrans[2], ptrans[2]],
        [0, ptrans[3], 1 - ptrans[3]]
    ])
    pi = np.array([1.0, 0.0, 0.0])
    for _ in range(50):
        pi = pi @ P
    avg_prob = pi[0] * pacc[0] + pi[1] * pacc[1] + pi[2] * pacc[2]
    mean_accidents_per_rider = 365 * avg_prob
    max_accidents = int(N * mean_accidents_per_rider * 1.5)
    batch_size = min(100000, max(10000, max_accidents))
    remb_cache = np.array(simule_remb(batch_size, params))
    for sim in range(n_simul):
        weather = simule_meteo(365, ptrans)
        probs = np.array([pacc[0] if state == "S" else pacc[1] if state == "N" else pacc[2] for state in weather])
        accidents_count = np.random.binomial(N, probs)
        total_accidents = np.sum(accidents_count)
        R_total = 0
        if total_accidents > 0:
            indices = np.random.choice(len(remb_cache), min(total_accidents, len(remb_cache)), 
                                      replace=(total_accidents > len(remb_cache)))
            R_total = np.sum(remb_cache[indices])
            if total_accidents > len(remb_cache):
                remaining = total_accidents - len(remb_cache)
                extra_remb = np.sum(simule_remb(remaining, params))
                R_total += extra_remb
        if R_total > s:
            values.append(R_total - s)
    if not values:
        return {"s": 0, "demi_largeur": 0}
    values = np.array(values)
    mean_value = np.mean(values)
    std_dev = np.std(values, ddof=1) if len(values) > 1 else 0
    demi_largeur = 1.96 * std_dev / np.sqrt(len(values))
    return {"s": mean_value, "demi_largeur": demi_largeur}

def msb(n_simul, params):
    s = params["s"]
    N = params["N"]
    pacc = params["pacc"]
    ptrans = params["ptrans"]
    values = []
    P = np.array([
        [1 - ptrans[0], ptrans[0], 0],
        [ptrans[1], 1 - ptrans[1] - ptrans[2], ptrans[2]],
        [0, ptrans[3], 1 - ptrans[3]]
    ])
    pi = np.array([1.0, 0.0, 0.0])
    for _ in range(50):
        pi = pi @ P
    avg_prob = pi[0] * pacc[0] + pi[1] * pacc[1] + pi[2] * pacc[2]
    mean_accidents_per_rider = 365 * avg_prob
    max_accidents = int(N * mean_accidents_per_rider * 1.5)
    batch_size = min(100000, max(10000, max_accidents))
    remb_cache = np.array(simule_remb(batch_size, params))
    for sim in range(n_simul):
        total_accidents = np.random.poisson(N * mean_accidents_per_rider)
        R_total = 0
        if total_accidents > 0:
            indices = np.random.choice(len(remb_cache), min(total_accidents, len(remb_cache)), 
                                      replace=(total_accidents > len(remb_cache)))
            R_total = np.sum(remb_cache[indices])
            if total_accidents > len(remb_cache):
                remaining = total_accidents - len(remb_cache)
                extra_remb = np.sum(simule_remb(remaining, params))
                R_total += extra_remb
        if R_total > s:
            values.append(R_total - s)
    if not values:
        return {"s": 0, "demi_largeur": 0}
    values = np.array(values)
    mean_value = np.mean(values)
    std_dev = np.std(values, ddof=1) if len(values) > 1 else 0
    demi_largeur = 1.96 * std_dev / np.sqrt(len(values))
    return {"s": mean_value, "demi_largeur": demi_largeur}
