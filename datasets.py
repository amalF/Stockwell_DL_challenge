import math
import os                                                                                                                                                                                                
import random                                                                                                                      

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

class SamplePipeline:
    """
    A class for that manages an input pipeline
    """

    def input_pipeline(self, filenames, batch_size, augmentation=False, evaluate=False):
        '''
        Function that uses the tensorflow API to read tfrecords from files, apply augmentation,
        shuffle, and batch the dataset.
        '''
        self.augmentation = augmentation
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        dataset = dataset.map(
                self._read_and_decode_tfrecords,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Only run through data once during evaluation
        if evaluate:
            num_repeat = 1
            batch_size = 1
        else:
            dataset = dataset.shuffle(buffer_size=1024)
            num_repeat = 1

        dataset = dataset.repeat(num_repeat)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)

        return dataset

    def _read_and_decode_tfrecords(self, record):
        """Function to read and decode tfrecords, that can be
        used by the dataset API.

        Args:
            - record: a single record from the TFReocrd

        Returns:
            - image, image_bw, label: the image and label tensors.
        """

        features = self._parse_single_example_from_tfrecord(record)
        # decode the binary string image data
        original, crop, label, norm_label, shelf_start, shelf_end, x_coords, y_coords = self._preprocess_features(features)
        return crop, norm_label 

    def _preprocess_features(self, features):
        """Function that returns preprocessed images.

        Args:
            - features: a dictionary like object of tensors that is returned
            from the parse operation.

        Returns:
            - image: an image tensor of shape [BATCH_SIZE, IMAGE_PIXELS].
            - label: a label tensor of shape [BATCH_SIZE, YY] with YY
            signifying the structure of our training value.
        """
        #  Decode image, set it to values between 0 and 1
        original = tf.cast(tf.image.decode_jpeg(features['image/original']), tf.float32) / 255.0
        original.set_shape([240, 320, 3])
        original = tf.reshape(original, [240, 320, 3])

        #  Decode image, set it to values between 0 and 1
        crop = tf.cast(tf.image.decode_jpeg(features['image/crop']), tf.float32) / 255.0
        crop.set_shape([200, 200, 3])
        crop = tf.reshape(crop, [200, 200, 3])

        # We augment images only when we are training
        if self.augmentation:
            original = tf.image.random_flip_left_right(original)
            original = self._apply_with_random_selector(original, lambda x,
                                                        ordering: self._distort_color(x, ordering),
                                                        num_cases=9)

            crop = tf.image.random_flip_left_right(crop)
            crop = self._apply_with_random_selector(crop, lambda x,
                                                    ordering: self._distort_color(x, ordering),
                                                    num_cases=9)

        original = tf.image.per_image_standardization(original)
        original = tf.reshape(original, [320*240*3])

        crop = tf.image.per_image_standardization(crop)
        #crop = tf.reshape(crop, [200*200*3])

        label = features['label/id']
        label = tf.cast(label, tf.float32)

        norm_label = features['label/norm_id']
        norm_label = tf.cast(norm_label, tf.float32)

        shelf_start = tf.cast(features['label/shelf_start'], tf.float32)
        shelf_end = tf.cast(features['label/shelf_end'], tf.float32)

        x_coords = features['label/x_coords'].values
        y_coords = features['label/y_coords'].values

        return original, crop, label, norm_label, shelf_start, shelf_end, x_coords, y_coords

    def _parse_single_example_from_tfrecord(self, value):
        """Parses a single example from the tfrecords we've written.  We only
        pull a subset ofthe available features.

        Args:
            - value: the value output of a TFRecordReader() object, which
            contains the raw serialized proto information for a TFRecord.

        Returns:
            - features: a dictionary like object of tensors that is returned
            from the parse operation.
        'image/crop_height': int64_feature(crop_height),
        'image/crop_width': int64_feature(crop_width),
        'image/original_height': int64_feature(height),
        'image/original_width': int64_feature(width),
        'image/depth': int64_feature(3),
        'image/mask': bytes_feature(mask_encoded),
        'image/crop': bytes_feature(crop_encoded),
        'image/original': bytes_feature(orig_encoded),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'label/feed': bytes_feature(feed.encode('utf8')),
        'label/product_name': bytes_feature(item_name.encode('utf8')),
        'label/shelf_start': float_feature(start),
        'label/shelf_end': float_feature(end),
        'label/id': int64_feature(item_id),
        'label/norm_id': int64_feature(norm_id),
        'label/x_coords': int64_list_feature(x_coords),
        'label/y_coords': int64_list_feature(y_coords),
        """
        feature_map = (
            tf.io.parse_single_example(value,
                                    features={'image/original': tf.io.FixedLenFeature([], tf.string),
                                              'image/crop': tf.io.FixedLenFeature([], tf.string),
                                              'label/id': tf.io.FixedLenFeature([1], tf.int64),
                                              'label/norm_id': tf.io.FixedLenFeature([1], tf.int64),
                                              'label/shelf_start': tf.io.FixedLenFeature([1], tf.float32),
                                              'label/shelf_end': tf.io.FixedLenFeature([1], tf.float32),
                                              'label/x_coords': tf.io.VarLenFeature(dtype=tf.int64),
                                              'label/y_coords': tf.io.VarLenFeature(dtype=tf.int64)
                                              })
            )
        return feature_map

    def _apply_with_random_selector(self, x, func, num_cases):
        """Computes func(x, sel), with sel sampled from [0...num_cases-1].
        Args:
            - x: input Tensor.
            - func: Python function to apply.
            - num_cases: Python int32, number of cases to sample sel from.
        Returns:
            - The result of func(x, sel), where func receives the value of the
            selector as a python integer, but sel is sampled dynamically.
        """
        sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
        # Pass the real x only to one of the func calls.
        return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

    def _distort_color(self, image, color_ordering=0, fast_mode=False,
                       scope=None):
        """Distort the color of a Tensor image.
        Each color distortion is non-commutative and thus ordering of the color ops
        matters. Ideally we would randomly permute the ordering of the color ops.
        Rather then adding that level of complication, we select a distinct ordering
        of color ops for each preprocessing thread.
        Args:
            - image: 3-D Tensor containing single image in [0, 1].
            - color_ordering: Python int, a type of distortion (valid values:
            0-3).
            - fast_mode: Avoids slower ops (random_hue and random_contrast)
            - scope: Optional scope for name_scope.
        Returns:
            - 3-D Tensor color-distorted image on range [0, 1]
        Raises:
            - ValueError: if color_ordering not in [0, 3]
        """
        with tf.name_scope('distort_color'):#(scope, 'distort_color', [image]):
            if fast_mode:
                if color_ordering == 0:
                    image = tf.image.random_brightness(image, max_delta=8. / 255.)
                    image = tf.image.random_saturation(image, lower=0.85, upper=1.1)
                else:
                    image = tf.image.random_saturation(image, lower=0.85, upper=1.1)
                    image = tf.image.random_brightness(image, max_delta=8. / 255.)
            else:
                if color_ordering == 0 or color_ordering == 4:
                    image = tf.image.random_brightness(image, max_delta=8. / 255.)
                    image = tf.image.random_saturation(image, lower=0.85, upper=1.1)
                    image = tf.image.random_hue(image, max_delta=0.05)
                    image = tf.image.random_contrast(image, lower=0.85, upper=1.1)
                elif color_ordering == 1 or color_ordering == 5:
                    image = tf.image.random_saturation(image, lower=0.85, upper=1.1)
                    image = tf.image.random_brightness(image, max_delta=8. / 255.)
                    image = tf.image.random_contrast(image, lower=0.85, upper=1.1)
                    image = tf.image.random_hue(image, max_delta=0.05)
                elif color_ordering == 2 or color_ordering == 6:
                    image = tf.image.random_contrast(image, lower=0.85, upper=1.1)
                    image = tf.image.random_hue(image, max_delta=0.05)
                    image = tf.image.random_brightness(image, max_delta=8. / 255.)
                    image = tf.image.random_saturation(image, lower=0.85, upper=1.1)
                elif color_ordering == 3 or color_ordering == 7:
                    image = tf.image.random_hue(image, max_delta=0.05)
                    image = tf.image.random_saturation(image, lower=0.85, upper=1.1)
                    image = tf.image.random_contrast(image, lower=0.85, upper=1.1)
                    image = tf.image.random_brightness(image, max_delta=8. / 255.)
                elif color_ordering == 8:
                    # Extra dark
                    darkness_level = random.uniform(-0.5, -0.95)
                    image = tf.image.adjust_brightness(image, delta=darkness_level)
                else:
                    raise ValueError('color_ordering must be in [0, 8]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


