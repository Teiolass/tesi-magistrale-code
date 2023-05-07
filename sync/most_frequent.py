import numpy as np
from collections import Counter
from mimic_loader import *

def most_frequent (patient: np.ndarray, k: int, n_codes: int) -> np.ndarray:
    counter = Counter()
    output  = np.zeros(patient.shape, dtype=patient.dtype)
    for i in range(len(patient)):
        counter.update(patient[i])
        counter.subtract(Counter({0:counter[0]}))
        for j, (v, c) in enumerate(counter.most_common(k)):
            if c == 0: break
            output[i,j] = v
    return output

def get_total_recall(prediction: np.ndarray, target: np.ndarray) -> float:
    total_recall = 0
    for i in range(len(target)):
        p = prediction[i]
        t = target[i]
        counter_p = Counter(p)
        counter_t = Counter(t)
        counter_p.subtract(Counter({0: counter_p[0]}))
        counter_t.subtract(Counter({0: counter_t[0]}))
        intersection = counter_p & counter_t
        recall = intersection.total() / counter_t.total()
        total_recall += recall
    return total_recall



if __name__ == '__main__':
    if not 'mimic' in globals() or not 'data' in globals():
        print('it seems that the variables `mimic` and `data` are not defined in the global namespace')
        print('I`m going to create them')
        mimic = Mimic.from_folder('/home/amarchetti/data/mimic-iii', '/home/amarchetti/data')
        data  = MimicData.from_mimic(mimic, pad_visits=False, pad_codes=True)
        print('data loaded')
    else:
        print('I have found the variables `mimic` and `data` in the global namespace')
        print('I think there is no need to recompute them')
    print()

    num_test = 3000
    n_codes  = data.get_num_codes()
    recall_param = [10, 20, 30]

    for k in recall_param:
        total_recall = 0
        num_recall   = 0
        for it in range(num_test):
            start = data.patients[it]
            end   = data.patients[it+1]
            input      = data.codes[start:end-1]
            output     = data.codes[start+1:end]
            prediction = most_frequent(input, k, n_codes)
            total_recall += get_total_recall(prediction, output)
            num_recall += len(input)
        recall = total_recall / num_recall
        print(f'recall@{k}: {100*recall:.2f}%')



