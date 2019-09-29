import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation


def get_affinity_mat(data):
    data = np.array(data)
    n = data.shape[0]
    xs = data[:, 0]
    mat = (-np.abs(xs[:, np.newaxis] - xs[np.newaxis, :])).astype(np.float)
    mat += np.eye(n) * np.median(mat)
    return mat


def AP(data, name="test.jpg", h=1280, w=960):
    M = get_affinity_mat(data)
    af = AffinityPropagation(max_iter=100, convergence_iter=30, affinity='precomputed').fit(M)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    c_list = [cluster_centers_indices[i] for i in labels]

    colors = ['red', 'blue', 'black', 'green', 'yellow', 'grey']
    plt.figure(figsize=(8, 6))
    plt.xlim([0, w])
    plt.ylim([h, 0])
    length = M.shape[0]
    centers = list(cluster_centers_indices)
    for i in range(length):
        d1 = data[i]
        d2 = data[c_list[i]]
        c = centers.index(c_list[i])
        # plt.plot([d2[0], d1[0]], [d2[1], d1[1]], color=colors[c], linewidth=1)
        if i == c_list[i]:
            plt.scatter((d1[0]), (d1[1]), color=colors[c], linewidth=3)
        else:
            plt.scatter((d1[0]), (d1[1]), color=colors[c], linewidth=1)
    plt.savefig(name.replace(".jpg", '_AP.jpg'))
    # print(c_list)

    return c_list


if __name__ == "__main__":
    data = [[956, 801], [159, 345], [941, 1086], [331, 809], [345, 673], [308, 976], [155, 640], [380, 385], [147, 736],
            [861, 1276], [956, 486], [163, 544], [375, 1278], [366, 518], [148, 427]]

    AP(data, name="test_ap.jpg")