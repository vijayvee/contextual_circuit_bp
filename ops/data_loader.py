import numpy as np
import tensorflow as tf


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def resize_image_label(im, model_input_image_size, f='bilinear'):
    """Resize images filter."""
    if f == 'bilinear':
        res_fun = tf.image.resize_images
    elif f == 'nearest':
        res_fun = tf.image.resize_nearest_neighbor
    elif f == 'bicubic':
        res_fun = tf.image.resize_bicubic
    elif f == 'area':
        res_fun = tf.image.resize_area
    else:
        raise NotImplementedError
    if len(im.get_shape()) > 3:
        # Spatiotemporal image set.
        nt = int(im.get_shape()[0])
        sims = tf.split(im, nt)
        for idx in range(len(sims)):
            # im = tf.squeeze(sims[idx])
            im = sims[idx]
            sims[idx] = res_fun(
                im,
                model_input_image_size)
        im = tf.squeeze(tf.stack(sims))
        if len(im.get_shape()) < 4:
            im = tf.expand_dims(im, axis=-1)
    else:
        im = res_fun(
            tf.expand_dims(im, axis=0),
            model_input_image_size)
        im = tf.squeeze(im, axis=0)
    return im


def crop_image_label(image, label, size, crop='random'):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    if len(size) > 2:
        size[-1] = image_shape[-1] + int(label.get_shape()[-1])
    if crop == 'random':
        combined = tf.concat([image, label], axis=-1)
        combined_crop = tf.random_crop(combined, size)
        image = combined_crop[:, :, :image_shape[-1]]
        label = combined_crop[:, :, image_shape[-1]:]
        return image, label
    else:
        # Center crop
        image = tf.image.resize_image_with_crop_or_pad(
            image,
            size[0],
            size[1])
        label = tf.image.resize_image_with_crop_or_pad(
            label,
            size[0],
            size[1])
        return image, label


def lr_flip_image_label(image, label):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    combined = tf.concat([image, label], axis=-1)
    combined_crop = tf.image.random_flip_left_right(combined)
    image = combined_crop[:, :, :image_shape[-1]]
    label = combined_crop[:, :, image_shape[-1]:]
    return image, label


def ud_flip_image_label(image, label):
    """Apply a crop to both image and label."""
    image_shape = [int(x) for x in image.get_shape()]
    combined = tf.concat([image, label], axis=-1)
    combined_crop = tf.image.random_flip_up_down(combined)
    image = combined_crop[:, :, :image_shape[-1]]
    label = combined_crop[:, :, image_shape[-1]:]
    return image, label


def random_crop(image, model_input_image_size):
    """Wrapper for random cropping."""
    im_size = image.get_shape().as_list()
    if len(im_size) == 3:
        return tf.random_crop(image, model_input_image_size)
    elif len(im_size) == 4:
        if im_size[-1] > 1:
            raise NotImplementedError
        crop_size = model_input_image_size[:2] + [im_size[0]]
        trans_image = tf.transpose(tf.squeeze(image), [1, 2, 0])
        crop_image = tf.expand_dims(
            tf.transpose(
                tf.random_crop(trans_image, crop_size),
                [2, 0, 1]), axis=-1)
        return crop_image
    else:
        raise NotImplementedError


def center_crop(image, model_input_image_size):
    """Wrapper for center crop."""
    im_size = image.get_shape().as_list()
    target_height = model_input_image_size[0]
    target_width = model_input_image_size[1]
    if len(im_size) == 3:
        return tf.image.resize_image_with_crop_or_pad(
            image,
            target_height=target_height,
            target_width=target_width)
    elif len(im_size) == 4:
        time_split_image = tf.split(image, im_size[0], axis=0)
        crops = []
        for idx in range(len(time_split_image)):
            it_crop = tf.image.resize_image_with_crop_or_pad(
                tf.squeeze(time_split_image[idx], axis=0),
                target_height=target_height,
                target_width=target_width)
            crops += [tf.expand_dims(it_crop, axis=0)]
        return tf.concat(crops, axis=0)
    else:
        raise NotImplementedError


def image_flip(image, direction):
    """Wrapper for image flips."""
    im_size = image.get_shape().as_list()
    if direction == 'left_right':
        flip_function = tf.image.random_flip_left_right
    elif direction == 'up_down':
        flip_function = tf.image.random_flip_up_down
    else:
        raise NotImplementedError

    if len(im_size) == 3:
        return flip_function(image)
    elif len(im_size) == 4:
        if im_size[-1] > 1:
            raise NotImplementedError
        trans_image = tf.transpose(tf.squeeze(image), [1, 2, 0])
        flip_image = tf.expand_dims(
            tf.transpose(flip_function(trans_image), [2, 0, 1]), axis=-1)
        return flip_image
    else:
        raise NotImplementedError


def image_augmentations(
        image,
        data_augmentations,
        model_input_image_size,
        label=None):
    """Coordinating image augmentations for both image and heatmap."""
    im_size = [int(x) for x in image.get_shape()]
    im_size_check = np.any(
        np.less_equal(
            model_input_image_size[:2],
            im_size[:2]))
    if data_augmentations is not None:
        if 'random_crop' in data_augmentations and im_size_check:
            image = random_crop(image, model_input_image_size)
            print 'Applying random crop.'
        if 'center_crop' in data_augmentations and im_size_check:
            image = center_crop(image, model_input_image_size)
            print 'Applying center crop.'
        if 'random_crop_image_label' in data_augmentations and im_size_check:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            image, label = crop_image_label(
                image=image,
                label=label,
                size=model_input_image_size,
                crop='random')
        if 'center_crop_image_label' in data_augmentations and im_size_check:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            image, label = crop_image_label(
                image=image,
                label=label,
                size=model_input_image_size,
                crop='center')
        if 'resize' in data_augmentations and im_size_check:
            if len(model_input_image_size) > 2:
                model_input_image_size = model_input_image_size[:2]
            image = resize_image_label(
                im=image,
                model_input_image_size=model_input_image_size,
                f='bilinear')
            print 'Applying bilinear resize.'
        if 'rotate' in data_augmentations and im_size_check:
            raise NotImplementedError
        if 'resize_nn' in data_augmentations and im_size_check:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            if len(model_input_image_size) > 2:
                model_input_image_size = model_input_image_size[:2]
            image = resize_image_label(
                im=image,
                model_input_image_size=model_input_image_size,
                f='nearest')
            print 'Applying nearest resize.'
        if 'resize_image_label' in data_augmentations and im_size_check:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            if len(model_input_image_size) > 2:
                model_input_image_size = model_input_image_size[:2]
            image = resize_image_label(
                im=image,
                model_input_image_size=model_input_image_size,
                f='bilinear')
            label = resize_image_label(
                im=label,
                model_input_image_size=model_input_image_size,
                f='bilinear')
            print 'Applying bilinear resize.'
        elif 'resize_nn_image_label' in data_augmentations and im_size_check:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            if len(model_input_image_size) > 2:
                model_input_image_size = model_input_image_size[:2]
            image = resize_image_label(
                im=image,
                model_input_image_size=model_input_image_size,
                f='nearest')
            label = resize_image_label(
                im=label,
                model_input_image_size=model_input_image_size,
                f='nearest')
            print 'Applying nearest resize.'
        else:
            pass
        if 'left_right' in data_augmentations:
            image = image_flip(image, direction='left_right')
            print 'Applying random flip left-right.'
        if 'up_down' in data_augmentations:
            image = image_flip(image, direction='up_down')
            print 'Applying random flip up-down.'
        if 'lr_flip_image_label' in data_augmentations:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            image, label = lr_flip_image_label(image, label)
        if 'ud_flip_image_label' in data_augmentations:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            image, label = ud_flip_image_label(image, label)
        if 'random_contrast' in data_augmentations:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            print 'Applying random contrast.'
        if 'random_brightness' in data_augmentations:
            assert len(image.get_shape()) == 3, '4D not implemented yet.'
            image = tf.image.random_brightness(image, max_delta=63.)
            print 'Applying random brightness.'
        if 'calculate_rate_time_crop' in data_augmentations:
            im_shape = image.get_shape().as_list()
            minval = im_shape[0] // 3
            time_crop = tf.random_uniform(
                [],
                minval=minval,
                maxval=im_shape[0],
                dtype=tf.int32)

            # For now always pull from the beginning
            indices = tf.range(0, time_crop, dtype=tf.int32)
            selected_image = tf.gather(image, indices)
            padded_image = tf.zeros(
                [im_shape[0] - time_crop] + im_shape[1:],
                dtype=selected_image.dtype)

            # Randomly concatenate pad to front or back
            image = tf.cond(
                pred=tf.greater(
                    tf.random_uniform(
                        [],
                        minval=0,
                        maxval=1,
                        dtype=tf.float32),
                    0.5),
                true_fn=lambda: tf.concat([selected_image, padded_image], axis=0),
                false_fn=lambda: tf.concat([padded_image, selected_image], axis=0)
            )
            image.set_shape(im_shape)

            # Convert label to rate
            label = label / im_shape[0]
        if 'calculate_rate' in data_augmentations:
            label = label / image.get_shape().as_list()[0]
    else:
        assert len(image.get_shape()) == 3, '4D not implemented yet.'
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_image_size[0], model_input_image_size[1])
    return image, label


def decode_data(features, reader_settings):
    """Decode data from TFrecords."""
    if features.dtype == tf.string:
        return tf.decode_raw(
            features,
            reader_settings)
    else:
        return tf.cast(
            features,
            reader_settings)


def read_and_decode(
        filename_queue,
        model_input_image_size,
        tf_dict,
        tf_reader_settings,
        data_augmentations,
        number_of_files,
        resize_output=None):
    """Read and decode tensors from tf_records and apply augmentations."""
    reader = tf.TFRecordReader()

    # Switch between single/multi-file reading
    if number_of_files == 1:
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features=tf_dict)
    else:
        _, serialized_examples = reader.read_up_to(
            filename_queue,
            num_records=number_of_files)
        features = tf.parse_example(
            serialized_examples,
            features=tf_dict)

    # Handle decoding of each element
    image = decode_data(
        features=features['image'],
        reader_settings=tf_reader_settings['image']['dtype'])
    label = decode_data(
        features=features['label'],
        reader_settings=tf_reader_settings['label']['dtype'])

    # Reshape each element
    image = tf.reshape(image, tf_reader_settings['image']['reshape'])
    if tf_reader_settings['label']['reshape'] is not None:
        label = tf.reshape(label, tf_reader_settings['label']['reshape'])

    if image.dtype == tf.float64:
        image = tf.cast(image,tf.float32)

    # Preprocess images and heatmaps
    if len(model_input_image_size) == 3:
        # 2D image augmentations
        image, label = image_augmentations(
            image=image,
            label=label,
            model_input_image_size=model_input_image_size,
            data_augmentations=data_augmentations)
        if resize_output is not None:
            # Resize labels after augmentations
            label = resize_image_label(
                im=label,
                model_input_image_size=resize_output,
                f='nearest')
    elif len(model_input_image_size) == 4:
        # 3D image augmentations.
        # TODO: optimize 3D augmentations with c++. This is slow.
        split_images = tf.split(
            image,
            model_input_image_size[0],
            axis=0)
        split_images = [tf.squeeze(im, axis=0) for im in split_images]
        images, labels = [], []
        if np.any(['label' in x for x in data_augmentations if x is not None]):
            split_labels = tf.split(
                label,
                model_input_image_size[0],
                axis=0)
            split_labels = [tf.squeeze(lab, axis=0) for lab in split_labels]
            for im, lab in zip(split_images, split_labels):
                it_im, it_lab = image_augmentations(
                    image=im,
                    label=lab,
                    model_input_image_size=model_input_image_size[1:],
                    data_augmentations=data_augmentations)
                if resize_output is not None:
                    # Resize labels after augmentations
                    it_lab = resize_image_label(
                        im=it_lab,
                        model_input_image_size=resize_output,
                        f='area')
                images += [it_im]
                labels += [it_lab]
            label = tf.stack(
                labels,
                axis=0)
            image = tf.stack(
                images,
                axis=0)
        else:
            if None not in data_augmentations:
                for im in split_images:
                    it_im = image_augmentations(
                        image=im,
                        model_input_image_size=model_input_image_size[1:],
                        data_augmentations=data_augmentations)
                    images += [it_im]
                image = tf.stack(
                    images,
                    axis=0)
    
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)
    return image, label


def inputs(
        dataset,
        batch_size,
        model_input_image_size,
        tf_dict,
        data_augmentations,
        num_epochs,
        tf_reader_settings,
        shuffle,
        number_of_files=1,
        resize_output=None):
    """Read tfrecords and prepare them for queueing."""
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 5 * batch_size
    num_threads = 2

    # Check if we need timecourses.
    if len(model_input_image_size) == 4:
        number_of_files = model_input_image_size[0]

    # Start data loader loop
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [dataset], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        batch_data = read_and_decode(
            filename_queue=filename_queue,
            model_input_image_size=model_input_image_size,
            tf_dict=tf_dict,
            tf_reader_settings=tf_reader_settings,
            data_augmentations=data_augmentations,
            number_of_files=number_of_files,
            resize_output=resize_output)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_after_dequeue)
        else:
            images, labels = tf.train.batch(
                batch_data,
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity)
        return images, labels
