import os
import sys
import importlib

import jax
import tensorflow as tf
import tensorflow_datasets as tfds


def _maybe_import_custom_builder(data_dir, dataset_name):
    if data_dir is None:
        return

    builders_root = os.path.join(os.path.dirname(os.path.abspath(data_dir)), 'tfds_builders')
    if not os.path.isdir(builders_root):
        return

    if builders_root not in sys.path:
        sys.path.insert(0, builders_root)

    module_candidates = [dataset_name]
    for dirpath, _, filenames in os.walk(builders_root):
        rel_dir = os.path.relpath(dirpath, builders_root)
        for filename in sorted(filenames):
            if not filename.endswith('.py') or filename == '__init__.py':
                continue
            module_stem = filename[:-3]
            if rel_dir == '.':
                module_name = module_stem
            else:
                module_name = rel_dir.replace(os.sep, '.') + '.' + module_stem
            module_candidates.append(module_name)

    for module_name in dict.fromkeys(module_candidates):
        try:
            importlib.import_module(module_name)
        except Exception:
            continue


def _find_built_dataset_dir(data_dir, dataset_name):
    if data_dir is None:
        return None
    dataset_root = os.path.join(data_dir, dataset_name)
    if not os.path.exists(dataset_root):
        return None

    candidates = []
    for dirpath, _, filenames in os.walk(dataset_root):
        if 'dataset_info.json' in filenames:
            candidates.append(dirpath)
    if not candidates:
        version_dirs = []
        for name in sorted(os.listdir(dataset_root)):
            candidate = os.path.join(dataset_root, name)
            if os.path.isdir(candidate):
                version_dirs.append(candidate)
        if version_dirs:
            return version_dirs[-1]
        return dataset_root
    candidates.sort(key=lambda path: (path.count(os.sep), path))
    return candidates[-1]


def _load_tfds_dataset(tfds_name, split, data_dir=None):
    _maybe_import_custom_builder(data_dir, tfds_name)
    try:
        return tfds.load(tfds_name, split=split, data_dir=data_dir, try_gcs=False)
    except Exception:
        built_dir = _find_built_dataset_dir(data_dir, tfds_name)
        if built_dir is None:
            raise
        builder = tfds.builder_from_directory(built_dir)
        return builder.as_dataset(split=split)


def _load_tfds_builder(tfds_name, data_dir=None):
    _maybe_import_custom_builder(data_dir, tfds_name)
    try:
        return tfds.builder(tfds_name, data_dir=data_dir, try_gcs=False)
    except Exception:
        built_dir = _find_built_dataset_dir(data_dir, tfds_name)
        if built_dir is None:
            raise
        return tfds.builder_from_directory(built_dir)


def _resolve_dataset(dataset_name):
    if 'imagenet256' in dataset_name:
        return 'imagenet2012', 'train', 'validation'
    if dataset_name == 'celebahq256':
        return 'celebahq256', 'train', 'train'
    if dataset_name == 'lsunchurch':
        return 'lsunc', 'church-train', 'church-test'
    raise ValueError(f"Unknown dataset {dataset_name}")


def get_num_examples(dataset_name, is_train, data_dir=None):
    tfds_name, train_split, valid_split = _resolve_dataset(dataset_name)
    split_name = train_split if is_train else valid_split
    builder = _load_tfds_builder(tfds_name, data_dir=data_dir)
    return builder.info.splits[split_name].num_examples


def get_dataset(
    dataset_name,
    batch_size,
    is_train,
    debug_overfit=False,
    data_dir=None,
    repeat=True,
):
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
            image = (image - 0.5) / 0.5
            return image, data['label']

        split_name = 'train' if (is_train or debug_overfit) else 'validation'
        if repeat:
            split = tfds.split_for_jax_process(split_name, drop_remainder=True)
        else:
            split = split_name
        dataset = _load_tfds_dataset('imagenet2012', split=split, data_dir=data_dir)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if debug_overfit:
            dataset = dataset.take(8)
            if repeat:
                dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
        else:
            if is_train:
                dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=repeat)
            if repeat:
                dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return iter(tfds.as_numpy(dataset))

    if dataset_name == 'celebahq256':
        def deserialization_fn(data):
            image = data['image']
            if is_train:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, data['label']

        split = 'train'
        dataset = _load_tfds_dataset('celebahq256', split=split, data_dir=data_dir)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if debug_overfit:
            dataset = dataset.take(8)
        elif is_train:
            dataset = dataset.shuffle(
                20000, seed=42 + jax.process_index(), reshuffle_each_iteration=repeat)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return iter(tfds.as_numpy(dataset))

    if dataset_name == 'lsunchurch':
        def deserialization_fn(data):
            image = data['image']
            min_side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
            image = tf.image.resize_with_crop_or_pad(image, min_side, min_side)
            image = tf.image.resize(image, (256, 256), antialias=True)
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - 0.5) / 0.5
            return image, 0

        split_name = 'church-train' if is_train else 'church-test'
        if repeat:
            split = tfds.split_for_jax_process(split_name, drop_remainder=True)
        else:
            split = split_name
        dataset = _load_tfds_dataset('lsunc', split=split, data_dir=data_dir)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            dataset = dataset.shuffle(10000, seed=42, reshuffle_each_iteration=repeat)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return iter(tfds.as_numpy(dataset))

    raise ValueError(f"Unknown dataset {dataset_name}")
