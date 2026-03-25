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
    elif dataset_name.startswith('celebahq256'):
        split_specs = {
            'train': 'train[:90%]',
            'validation': 'train[90%:95%]',
            'test': 'train[95%:]',
        }
        if dataset_name == 'celebahq256_valid':
            split_name = 'validation'
        elif dataset_name == 'celebahq256_test':
            split_name = 'test'
        else:
            split_name = 'train' if (is_train or debug_overfit) else 'validation'
        apply_augmentation = split_name == 'train' and is_train

        def deserialization_fn(data):
            image = data['image']
            if apply_augmentation:
                image = tf.image.random_flip_left_right(image)
            image = tf.cast(image, tf.float32)
            image = image / 255.0
            image = (image - 0.5) / 0.5 # Normalize to [-1, 1]
            return image,  data['label']

        # CelebA-HQ is loaded from a TFDS builder without a dedicated validation/test
        # stream in this codepath, so we carve out deterministic holdout slices from
        # the train split: 90% train, 5% validation, 5% test.
        split = split_specs[split_name]
        dataset = tfds.load('celebahq256', split=split)
        dataset = dataset.map(deserialization_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if debug_overfit:
            dataset = dataset.take(8)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
        elif split_name == 'train':
            dataset = dataset.shuffle(20000, seed=42+jax.process_index(), reshuffle_each_iteration=True)
            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        else:
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
