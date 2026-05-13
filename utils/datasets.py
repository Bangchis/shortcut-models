import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import jax
import os


def _candidate_data_dirs(data_dir):
    if data_dir is None:
        return []
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    return [
        data_dir,
        os.path.join(data_dir, 'tensorflow_datasets'),
        os.path.join(data_dir, 'data', 'tensorflow_datasets'),
        os.path.join(data_dir, 'data'),
    ]


def _has_dataset_info(path):
    return os.path.exists(os.path.join(path, 'dataset_info.json'))


def _dataset_info_dirs(base_dir, max_depth=5):
    if not os.path.isdir(base_dir):
        return []
    base_depth = base_dir.rstrip(os.sep).count(os.sep)
    matches = []
    for root, dirs, files in os.walk(base_dir):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if 'dataset_info.json' in files:
            matches.append(root)
            dirs[:] = []
    return matches


def _find_builder_dir(data_dir, dataset_names):
    dataset_names = tuple(dataset_names)
    for base_dir in _candidate_data_dirs(data_dir):
        if _has_dataset_info(base_dir):
            return base_dir
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(base_dir, dataset_name)
            if _has_dataset_info(dataset_dir):
                return dataset_dir
            if not os.path.isdir(dataset_dir):
                continue
            for version in sorted(os.listdir(dataset_dir)):
                version_dir = os.path.join(dataset_dir, version)
                if os.path.isdir(version_dir) and _has_dataset_info(version_dir):
                    return version_dir
        matches = _dataset_info_dirs(base_dir)
        named_matches = [
            path for path in matches
            if any(name in path.split(os.sep) for name in dataset_names)
        ]
        if named_matches:
            return sorted(named_matches)[0]
        if len(matches) == 1:
            return matches[0]
    return None


def load_tfds_split(tfds_name, split, data_dir=None, aliases=()):
    builder_dir = _find_builder_dir(data_dir, (tfds_name, *aliases))
    if builder_dir is not None:
        print(f'Loading TFDS builder from {builder_dir}')
        return tfds.builder_from_directory(builder_dir).as_dataset(split=split)
    if data_dir is not None:
        print(f'No TFDS builder directory found under {data_dir}; falling back to tfds.load({tfds_name})')
    return tfds.load(tfds_name, split=split, data_dir=data_dir)


def get_dataset(dataset_name, batch_size, is_train, debug_overfit=False, data_dir=None):
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
        dataset = load_tfds_split('imagenet2012', split, data_dir=data_dir)
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
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image,  data['label']

        split = tfds.split_for_jax_process('train' if (is_train or debug_overfit) else 'validation', drop_remainder=True)
        dataset = load_tfds_split(
            'celebahq256', split, data_dir=data_dir, aliases=('celeb_a_hq',))
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
        dataset = load_tfds_split('lsunc', split, data_dir=data_dir)
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
