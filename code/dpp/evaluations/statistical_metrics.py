import numpy as np

def distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def travel_distance(geo):
    return np.sum(distance(geo[:-1, 0], geo[:-1, 1], geo[1:, 0], geo[1:, 1]))


def radius(geo):
    center = np.mean(geo, axis = 0)
    return np.sqrt(np.mean(distance(geo[:, 0], geo[:, 1], center[0], center[1])))


def JSD(P_A, P_B):
    epsilon = 1e-14
    P_A = (P_A / P_A.sum() + epsilon)
    P_B = (P_B / P_B.sum() + epsilon)
    P_merged = 0.5 * (P_A + P_B)
    
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    jsd = 0.5 * (kl_PA_PM + kl_PB_PM)
    return jsd

def arr_to_distribution(arr, min, max, bins):
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution

def compute_probability_distribution(data):
    unique_elements, counts = np.unique(data, return_counts=True)
    total_counts = np.sum(counts)
    probabilities = counts / total_counts
    return unique_elements, probabilities

def category_jsd(generated_category, real_category):
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)

    p,q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p = np.asarray(p)
    q = np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p:
        p_probs[element] = prob
    
    for element, prob in q:
        q_probs[element] = prob

    jsd_value = JSD(np.array(list(p_probs.values())),np.array(list(q_probs.values())))
    return jsd_value


def evaluation(generated, original):
    generated = np.array(generated)
    original = np.array(original)
    max = np.max(generated) if np.max(generated) > np.max(original) else np.max(original)
    p_gen = arr_to_distribution(generated, 0, max, 100)
    p_real = arr_to_distribution(original, 0, max, 100)
    jsd = JSD(p_gen, p_real)
    return jsd

def get_visits(trajs,max_locs):
    visits = np.zeros(shape=(max_locs), dtype=float)
    for t in trajs:
        visits[t] += 1
    visits = visits / np.sum(visits)
    return visits

def get_topk_visits(visits, K):
    locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
    locs_visits.sort(reverse=True, key=lambda d: d[1])
    topk_locs = [locs_visits[i][0] for i in range(K)]
    topk_probs = [locs_visits[i][1] for i in range(K)]
    return np.array(topk_probs), topk_locs

def Get_Statistical_Metrics(real_data, generated_data, min_seq_len=1):
    Real_Statistics = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category':[]}
    Generated_Statistics = {'Distance': [], 'Radius': [], 'DailyLoc': [], 'Interval': [], 'Category':[]}
    JSD = {'Distance': 1.0, 'Radius': 1.0, 'DailyLoc': 1.0, 'Interval': 1.0, 'Category':1.0}
    data = [generated_data,real_data]
    assert len(generated_data) > 0
    assert len(real_data) > 0
    metrics_dicts = [Generated_Statistics,Real_Statistics]
    for idx,seqs in enumerate(data):
        for seq in seqs:
            if len(seq['checkins'])>=1:
                gps = np.array(seq['gps'])
                metrics_dicts[idx]['Distance'].append(travel_distance(gps))
                metrics_dicts[idx]['Radius'].append(radius(gps))
                metrics_dicts[idx]['DailyLoc'].append(len(set(seq['checkins'])))
                metrics_dicts[idx]['Interval'].extend(np.ediff1d(np.concatenate([[seq["t_start"]], seq["arrival_times"], [seq["t_end"]]])).tolist())
                metrics_dicts[idx]['Category'].extend(seq['marks'])

    for metric in JSD.keys():
        if metric == 'Category':
            JSD[metric] = category_jsd(Generated_Statistics[metric],Real_Statistics[metric])
        else :
            JSD[metric] = evaluation(Generated_Statistics[metric],Real_Statistics[metric])
    return JSD