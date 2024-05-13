import torch
from torch_cluster import knn
from sortedcontainers import SortedDict
import bisect


class LSHash(object):
    """ LSHash implements locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_bits:
        Number of bits to use in each hash function.（超平面的数量）
        The hash will output integer values in [0, 2**num_bits - 1].

    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900. d
    :param num_hashtables 这里naive的我先当成1来实现了:
        (optional) The number of hash tables used for multiple lookups.
        the num of hyperplane
    """

    def __init__(self, input_dim, hash_bits, N, num_hashtables=1):
        self.hash_bits = hash_bits
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.N = N
        self.L = int(hash_bits / N)
        # self.powers_of_two = torch.tensor([2 ** i for i in range(self.hash_bits)], dtype=torch.float32)
        # self.projections = self._init_random_planes()
        self.projections = self._init_orthogonal_planes(self.input_dim, self.N, self.L)
        # self.hashtables = dict()
        self.hashtables = SortedDict()
        self.maxnum = 50000

    def _init_random_planes(self):
        """ Initialize random planes used to calculate the hashes
        plane矩阵的形状：n_bits * input_dim
            self.__dict__ 返回一个包含实例所有属性和方法的字典
        """

        # 检查对象实例是否已经具有 "random_planes" 这个属性。如果已经存在，说明已经初始化过了，函数直接返回
        if "random_planes" in self.__dict__:
            return

        return torch.randn(self.input_dim, self.hash_bits)

    def _init_orthogonal_planes(self, d, N, L):
        # Initialize H matrix
        H = torch.randn(d, N * L)
        H_reshaped = H.view(d, N, L)

        # Initialize output matrix，input_dim*hashbits
        output = torch.zeros(d, N * L)

        for i in range(L):
            # Select the ith batch of H
            Hi = H_reshaped[:, :, i]

            # Compute QR decomposition
            # Q.shape : (d,N)
            Q, R = torch.linalg.qr(Hi, mode="complete")
            # Take the first N columns of Q
            Wi = Q[:, :N]

            # Store Wi in the output matrix
            output[:, i * N: (i + 1) * N] = Wi

        return output

    # def _init_hashtables(self):
    #     """ Initialize the hash tables such that each record will be in the
    #     form of "[dict1, dict2, ...]" """
    #     self.hash_tables = [dict() for _ in range(self.num_hashtables)]

    def hash(self, x):
        """Calculates hash codes for input x.

        Arguments:
            x: Tensor shape [batch_size, dimensions]
            or shape [1,dimensions].

        Returns:
            Tensor of [batch_size, 1] hash codes
            of type tf.int64. Hash codes are in range [0, 2**num_bits - 1].
        """
        # projs = torch.tensordot(x, self.projections, dims=([-1], [-1]))
        projs = torch.tensordot(x, self.projections, dims=([-1], [0]))
        # projs.shape: [batch_size, hash_bits]
        signs = torch.clamp(torch.sign(projs), 0.0, 1.0).int()
        # signs.shape: [batch_size,  hash_bits]
        hash_strings = [''.join(map(str, row.tolist())) for row in signs]
        return hash_strings

    def index(self, x):
        """
        根据hash得到的hashcode将每个point分桶
        """
        hash_strings = self.hash(x)  # !!计算hash

        for i, value in enumerate(hash_strings):
            if value not in self.hashtables:
                self.hashtables[value] = [i]
            else:
                self.hashtables[value].append(i)

    def find_closest_keys(self, keys, x, index, max_distance):
        closest_keys = []
        i = index - 1
        j = index

        # Check keys with smaller and greater indexes
        while i >= 0 or j < len(keys):
            if i >= 0 and self.calculate_distance_key(x, keys[i], "hamming") == max_distance:
                closest_keys.append(keys[i])
            if j < len(keys) and self.calculate_distance_key(x, keys[j], "hamming") == max_distance:
                closest_keys.append(keys[j])
            i -= 1
            j += 1

        return closest_keys

    def batch_query(self, xb, query_points, k=None):
        """
      Arguments:
          :param xb: basic dataset
          :param n: Number of vectors in the batch.
          :param query_points: Tensor of input vectors to search, size [n, dimensions].
          :param k: Number of extracted vectors.
          
      """
        indices_list = []

        # 计算查询点的哈希值 batch_size,stringlen
        query_hashes = self.hash(query_points)

        # 建立一个字典，用于存储每个哈希值对应的查询点索引
        query_buckets = {}

        for i, hash_val in enumerate(query_hashes):
            if hash_val not in query_buckets:
                query_buckets[hash_val] = [i]
            else:
                query_buckets[hash_val].append(i)
        # 同一个桶内的tensor在query_points里的序号：in_bucket_idx
        in_bucket_idx = torch.cat([torch.tensor(sublist) for sublist in query_buckets.values()], dim=0)

        max_bucket_size = 1000

        for hash_val, query_indices in query_buckets.items():
            # 哈希值相同的查询点一起查询
            bucket_xq = query_points[query_indices].cuda()
            # bucket_xq = query_points[query_indices]
            # print(bucket_xq.shape)

            # 获取属于当前哈希值对应的桶内的点
            bucket_indices = []
            if hash_val in self.hashtables.keys():
                bucket_indices.extend(self.hashtables.get(hash_val, []))  # 是一个list

            distance = 1
            keys = list(self.hashtables.keys())
            lbound = bisect.bisect_left(keys, hash_val)
            while len(bucket_indices) < max_bucket_size:
                closest_keys = self.find_closest_keys(keys, hash_val, lbound, distance)
                for key in closest_keys:
                    bucket_indices.extend(self.hashtables.get(key, []))
                    if len(bucket_indices) >= max_bucket_size:
                        break
                distance += 1

            bucket_xb = xb[bucket_indices].cuda()

            # bucket_xb = xb[bucket_indices]
            # print(bucket_xb.shape)

            knn_indices = knn(bucket_xb, bucket_xq, k)

            knn_indice = knn_indices[1]
            original_indices = [bucket_indices[i] for i in knn_indice]
            original_indices = torch.tensor(original_indices, dtype=torch.int)

            indices_list.append(original_indices)

        batch_size = query_points.shape[0]
        indices = torch.cat(indices_list, dim=0).reshape(batch_size, k)

        result = torch.full((batch_size, k), -1, dtype=torch.long)

        # for i, idx in enumerate(in_bucket_idx):
        #     result[idx] = indices[i]
        expanded_in_bucket_idx = in_bucket_idx.unsqueeze(1).expand(batch_size, k)

        # 使用 scatter 函数将 indices 根据 in_bucket_idx 分散到 result 中
        result.scatter_(dim=0, index=expanded_in_bucket_idx, src=indices)
        return result

    def query2(self, x, query_point, k=None, distance_func=None):
        """
        Arguments:
            :param query_point: Tensor shape [1,dimensions].
            :param k:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
            :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean")
        """
        candidates = set()
        query_hash = self.hash(query_point)

        # 遍历哈希表，找到与查询哈希值最接近的键
        closest_key = min(self.hashtables.keys(),
                          key=lambda key: self._calculate_distance_key(query_hash, key, distance_func))

        candidates.update(self.hashtables.get(closest_key, []))
        # 这里还要对candidate进行排序
        """
        rank candidates by distance function
       
        """
        # rank candidates by distance function
        candidates = [(ix, self._calculate_distance(query_point, x[ix]))
                      for ix in candidates]
        candidates.sort(key=lambda x: x[1])
        return candidates[:k] if k else candidates

    def _calculate_distance(self, x1, x2):
        """
        计算两个数据点之间的距离。
        """
        # 欧几里得距离
        return torch.norm(x1 - x2, p=2).item()
        # else:
        #     raise ValueError("Unsupported distance function. Supported functions are 'hamming' and 'euclidean'.")

    def calculate_distance_key(self, hash1, hash2, distance_func):
        """
        计算两个哈希值之间的距离。
        """
        if distance_func == "hamming":
            # 汉明距离
            # xor_result = hash1 ^ hash2
            # # 统计异或结果中1的个数，即汉明距离
            # distance = bin(xor_result).count('1')
            distance = sum(1 for x, y in zip(hash1, hash2) if x != y)
            return distance
        elif distance_func == "euclidean":
            # 欧几里得距离
            return ((hash1 - hash2) ** 2) ** 0.5
        else:
            raise ValueError("Unsupported distance function. Supported functions are 'hamming' and 'euclidean'.")

    def calculate_recall(self, true_neighbors, retrieved_neighbors, k):
        assert k > 0

        retrieved_neighbors = retrieved_neighbors[:k]

        # 计算交集的元素个数
        intersection_count = torch.sum(torch.isin(retrieved_neighbors, true_neighbors))

        # 计算recall值
        recall_value = intersection_count.item() / k

        return recall_value

    def batch_get_recall(self, true_neighbors, retrieved_neighbors, k):
        """
      Calculate recall for a batch of queries.

      Arguments:
          :param true_neighbors: True neighbors for each query in the batch, shape [batch_size, m].
          :param retrieved_neighbors: Retrieved neighbors for each query in the batch, shape [batch_size, m].
          :param k: Number of neighbors to consider for recall calculation.
      """
        assert k > 0

        # 初始化一个列表用于存储每个查询的交集数量
        intersection_counts = []

        # 逐行处理每个查询的邻居
        for true_neighbor, retrieved_neighbor in zip(true_neighbors, retrieved_neighbors):
            # 获取每个查询的前 k 个检索到的邻居
            retrieved_neighbor = retrieved_neighbor[:k]
            true_neighbor = true_neighbor[:k]
            # 计算逐元素的交集的元素个数
            intersection_count = torch.sum(torch.isin(retrieved_neighbor, true_neighbor))

            # 将结果添加到列表中
            intersection_counts.append(intersection_count.item())

        # 将列表转换为张量
        intersection_counts = torch.tensor(intersection_counts)

        # 计算recall值，每个查询的 recall
        recall_values = intersection_counts.float() / k

        # 计算平均 recall
        average_recall = torch.mean(recall_values)

        return recall_values, average_recall
