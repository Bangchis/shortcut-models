import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax

def get_dataset(dataset_name, batch_size, is_train, debug_overfit=False):
    print("Loading dataset")
    if 'imagenet256' in dataset_name:
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            if 'imagenet256' in dataset_name:
                image = tf.image.resize(image, (256, 256), antialias=True)
            elif 'imagenet128' in dataset_name:
                image = tf.image.resize(image, (256, 256), antialias=True)
            else:
                raise ValueError(f"Unknown dataset {dataset_name}")
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image, data['label']

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else 'validation', drop_remainder=True)
        dataset = tfds.load('imagenet2012', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if debug_overfit:
            dataset = dataset.take(8)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'celebahq256':
        def deserialization_fn(data):
            image = data['image']
            image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image,  data['label']

        # split = tfds.split_for_jax_process('train' if is_train else 'validation', drop_remainder=True)
        split='train'
        dataset = tfds.load('celebahq256', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(20000, seed=42+jax.process_index(), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    elif dataset_name == 'lsunchurch':
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image, 0 # No label

        split = tfds.split_for_jax_process('church-train' if is_train else 'church-test', drop_remainder=True)
        dataset = tfds.load('lsunc', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        dataset = iter(dataset)
        return dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_random_access_dataset(dataset_name, is_train=True):
    """Returns a tfds ArrayRecordDataSource supporting source[i] indexing.
    Used for within-cluster data sampling in Stage C.
    """
    if 'imagenet256' in dataset_name or 'imagenet128' in dataset_name:
        split = 'train' if is_train else 'validation'
        return tfds.data_source('imagenet2012', split=split)
    elif dataset_name == 'celebahq256':
        split = 'train' if is_train else 'test'
        return tfds.data_source('celebahq256', split=split)
    elif dataset_name == 'lsunchurch':
        split = 'church-train' if is_train else 'church-test'
        return tfds.data_source('lsunc', split=split)
    else:
        raise ValueError(f"get_random_access_dataset: Unknown dataset {dataset_name}")


def get_ordered_dataset(dataset_name, batch_size):
    """Returns a non-shuffled ordered dataset iterator for cluster precomputation.
    No shuffle, no repeat; suitable for one-pass cluster assignment.
    Returns (iterator, None).
    """
    if 'imagenet256' in dataset_name or 'imagenet128' in dataset_name:
        target_size = 128 if 'imagenet128' in dataset_name else 256

        def deserialization_fn_no_flip(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (target_size, target_size), antialias=True)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        split = tfds.split_for_jax_process('train', drop_remainder=False)
        dataset = tfds.load('imagenet2012', split=split)
        dataset = dataset.map(deserialization_fn_no_flip, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        return iter(dataset), None
    elif dataset_name == 'celebahq256':
        def deserialization_fn_no_flip(data):
            image = data['image']
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        dataset = tfds.load('celebahq256', split='train')
        dataset = dataset.map(deserialization_fn_no_flip, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        return iter(dataset), None
    elif dataset_name == 'lsunchurch':
        def deserialization_fn_no_flip(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, 0

        split = tfds.split_for_jax_process('church-train', drop_remainder=False)
        dataset = tfds.load('lsunc', split=split)
        dataset = dataset.map(deserialization_fn_no_flip, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)
        return iter(dataset), None
    else:
        raise ValueError(f"get_ordered_dataset: Unknown dataset {dataset_name}")


def preprocess_images_from_source(raw_records, dataset_name, training=True):
    """Apply the same preprocessing as get_dataset() to raw tfds records.
    raw_records: list of dicts with 'image' key (uint8 numpy [H, W, 3]).
    Returns: (images [B, H, W, 3] float32 in [-1,1], labels [B] int32).
    """
    target_size = 128 if 'imagenet128' in dataset_name else 256
    processed_images = []
    processed_labels = []
    for rec in raw_records:
        img = rec['image']  # uint8 [H, W, 3]
        label = int(rec['label']) if 'label' in rec else 0
        h, w = img.shape[0], img.shape[1]
        min_side = min(h, w)
        img_tf = tf.image.resize_with_crop_or_pad(img, min_side, min_side)
        img_tf = tf.image.resize(img_tf, (target_size, target_size), antialias=True)
        if training and ('imagenet256' in dataset_name or 'imagenet128' in dataset_name):
            img_tf = tf.image.random_flip_left_right(img_tf)
        img_np = tf.cast(img_tf, tf.float32).numpy() / 255.0
        img_np = (img_np - 0.5) / 0.5
        processed_images.append(img_np)
        processed_labels.append(label)
    return np.stack(processed_images, axis=0), np.array(processed_labels, dtype=np.int32)


def sample_cluster_batch(cluster_indices, cluster_sizes, batch_size,
                          random_access_source, dataset_name,
                          pi_np=None, rng=None):
    """HOST-SIDE ONLY. Sample a batch where each sample (x0, x1) shares a cluster.

    cluster_indices: list of K np.arrays of global image indices
    cluster_sizes:   np.array [K] int64
    batch_size:      B
    random_access_source: tfds ArrayRecordDataSource supporting source[i]
    dataset_name:    for preprocessing
    pi_np:           optional [K] mixture weights (normalized internally); None = uniform
    rng:             optional numpy rng; defaults to np.random

    Returns:
        x1_images: np.ndarray [B, H, W, 3] float32 in [-1, 1]
        x1_labels: np.ndarray [B] int32
        k_batch:   np.ndarray [B] int32 cluster indices
    """
    K = len(cluster_indices)
    rng = rng if rng is not None else np.random

    if pi_np is not None:
        pi_np = np.asarray(pi_np, dtype=np.float64)
        pi_np = pi_np / pi_np.sum()
        k_batch = rng.choice(K, size=batch_size, p=pi_np)
    else:
        k_batch = rng.randint(0, K, size=batch_size)

    total_size = int(np.sum(cluster_sizes))
    global_indices = []
    for b in range(batch_size):
        k = int(k_batch[b])
        c_size = int(cluster_sizes[k])
        if c_size == 0:
            global_indices.append(int(rng.randint(0, max(total_size, 1))))
        else:
            local_idx = int(rng.randint(0, c_size))
            global_indices.append(int(cluster_indices[k][local_idx]))

    raw_records = [random_access_source[i] for i in global_indices]
    x1_images, x1_labels = preprocess_images_from_source(
        raw_records, dataset_name, training=True)
    return x1_images, x1_labels, k_batch.astype(np.int32)