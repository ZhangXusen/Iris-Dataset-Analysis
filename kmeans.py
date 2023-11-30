import numpy as np
def import_data_from_iris(filename):
    data = []
    cluster_raw = []

    with open(str(filename), 'r') as f:
        for line in f:
            line_temp = line.strip().split()
            line_temp_dumy = []
            for j in range(0, len(line_temp) - 1):
                line_temp_dumy.append(float(line_temp[j]))
            data.append(line_temp_dumy)
            cluster_raw.append(line_temp[j + 1])

    return data, cluster_raw


def eucl_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def init_centroids(data, k):
    samples_num, dim = data.shape
    num_arr = np.arange(0, samples_num)
    np.random.shuffle(num_arr)
    centroids = data[num_arr[:k], :]
    return centroids


def k_means(data, k):
    samples_num = data.shape[0]
    cluster_data = np.array(np.zeros((samples_num, 2)))
    cluster_changed = True
    centroids = init_centroids(data, k)
    print("初始类中心点：\n", centroids)
    count = 0

    while cluster_changed:
        count += 1
        cluster_changed = False
        for i in range(samples_num):
            min_dist = 100000.0
            min_index = 0
            for j in range(k):
                distance = eucl_distance(centroids[j, :], data[i, :])
                if distance < min_dist:
                    min_dist = distance
                    cluster_data[i, 1] = min_dist
                    min_index = j
            if cluster_data[i, 0] != min_index:
                cluster_changed = True
                cluster_data[i, 0] = min_index
        for j in range(k):
            cluster_index = np.nonzero(cluster_data[:, 0] == j)
            points_in_cluster = data[cluster_index]
            centroids[j, :] = np.mean(points_in_cluster, axis=0)

    print("迭代次数：", count)
    return centroids, cluster_data


def calculate_accuracy(cluster_data, k_num):
    right = 0
    for k in range(0, k_num):
        checker = [0, 0, 0]
        for i in range(0, 50):
            checker[int(cluster_data[i + 50 * k, 0])] += 1
        right += max(checker)
    return right


if __name__ == '__main__':
    data, cluster_raw = import_data_from_iris("iris.data")
    print("iris.dat数据：\n", data)
    dataArr = np.array(data)

    centroids, cluster_data = k_means(dataArr, 3)
    print("聚类结果类中心：\n", centroids)

    for i in range(3):
        print(i, "类包含的样本")
        for j in range(len(data)):
            if int(cluster_data[j, 0]) == i:
                print(j, data[j], "到质心距离：", cluster_data[j, 1])

    right_num = calculate_accuracy(cluster_data, 3)
    print("错误率：", 1 - right_num / len(data))

