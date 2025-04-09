import tensorflow as tf

def label_based_partition(y_true, y_pred):
    # 함수 내용
    labels = tf.argmax(y_true, axis=-1)  # 원핫 인코딩 제거.
    y_pred_partitions = []
    for label in range(10):  
        mask = tf.equal(labels, label)
        partition = tf.boolean_mask(y_pred, mask)
        y_pred_partitions.append(partition)
    return y_pred_partitions


@tf.function
def k_means_clustering(data, k=2, max_iterations=100):

    num_points = tf.shape(data)[0]
    dimensions = tf.shape(data)[1]

    centroids = tf.slice(tf.random.shuffle(data), [0, 0], [k, dimensions]) # 배치 사이즈가 작으면 k개를 못뽑는 불상사가 생김.

    old_centroids = tf.zeros([k, dimensions])

    def should_continue(i, centroids, old_centroids):
        return i<max_iterations

    def iteration(i, centroids, old_centroids):
        expanded_vectors = tf.expand_dims(data, 0)
        expanded_centroids = tf.expand_dims(centroids, 1)

        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
        assignments = tf.argmin(distances, 0)

        means = []
        for c in range(k):
            means.append(tf.reduce_mean(
                tf.gather(data,
                          tf.reshape(
                              tf.where(tf.equal(assignments, c)),
                              [1, -1])
                          ), axis=[1]))

        new_centroids = tf.concat(means, 0)
        return [i + 1, new_centroids, centroids]

    _, centroids, _ = tf.while_loop(should_continue, iteration, [0, centroids, old_centroids])

    return centroids


@tf.function
def kmeans_cluster_loss(y_true, y_pred):
    y_pred_partitions = label_based_partition(y_true, y_pred)
    sum_similarity = 0
    cent_list = [] #
    for label_tens in y_pred_partitions :
        cents = k_means_clustering(label_tens)
        label_cent=tf.reduce_sum(cents,axis = 0)/2 # 열끼리 싹 더함.
        cent_list.append(label_cent)

        cents=tf.nn.l2_normalize(cents, axis=1) # 코사인 유사도 계산
        similarity = tf.tensordot(cents[0], cents[1],axes=1)
        sum_similarity += tf.reduce_sum(similarity) # 코사인 유사도 전부 더해서 로그 씌울것.
        # tf.print("similarity")
        # tf.print(similarity, sum_similarity)  
        
    cent_concated = tf.stack(cent_list, axis=0) # 2차원 텐서 생성  
    label_distance = tf.norm(
        tf.expand_dims(cent_concated, axis=1)
        - tf.expand_dims(cent_concated, axis=0),
        axis=-1,
    )
    #tf.print("label dis : ")
    #tf.print(label_distance)
    label_distance = tf.reduce_sum(cent_concated)

   
    crossentropy = tf.keras.losses.CategoricalCrossentropy()

    return (
        crossentropy(y_true, y_pred) 
        - 0.1 * tf.keras.backend.log(1+ sum_similarity) # sum_similarity 는 클 수록 좋은것. 최대 10임. 로스는 나쁜거니까 뺴줘야. 0.1 안해줘도 학습 잘 함.
        - 0.1 * tf.keras.backend.log(1 + label_distance) # label_distance 도 클 수록 좋은것. 라벨과의 거리가 멀어졌다는걸 의미.
    )