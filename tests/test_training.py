import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from scipy import misc
from tqdm import tqdm


def image_batcher(
        start,
        num_batches,
        images,
        labels,
        batch_size):
    """Placeholder image/label batch loader."""
    for b in range(num_batches):
        next_image_batch = images[start:start + batch_size]
        image_stack = []
        label_stack = labels[start:start + batch_size]
        for f in next_image_batch:
            # 1. Load image patch
            patch = misc.imread(f)
            if len(patch.shape) == 2:
                image_stack += [patch[None, :, :, None]]
            else:
                image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += batch_size
        yield np.concatenate(image_stack, axis=0), label_stack, next_image_batch


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def training_loop(
        config,
        db,
        sess,
        train_op,
        summary_op,
        summary_writer,
        train_loss,
        val_loss,
        saver,
        dataset_module,
        summary_dir,
        checkpoint_dir,
        val_accuracy,
        train_accuracy,
        train_images,
        train_labels,
        val_images,
        val_labels):
    step, time_elapsed = 0, 0
    train_losses, train_accs, val_losses, val_accs, timesteps = {}, {}, {}, {}, {}
    files, labels = dataset_module.get_data()
    combined_files = files['train']
    combined_labels = labels['train']
    batch_size = config.batch_size
    num_batches = len(combined_files) // batch_size
    try:
        for image_batch, label_batch, _ in tqdm(
                image_batcher(
                    start=0,
                    num_batches=num_batches,
                    images=combined_files,
                    labels=combined_labels,
                    batch_size=batch_size),
                total=num_batches):
            feed_dict = {
                train_images: image_batch.astype(np.float32),
                train_labels: label_batch
            }
            start_time = time.time()
            _, loss_value, train_acc = sess.run(
                [
                    train_op,
                    train_loss,
                    train_accuracy
                ],
                feed_dict=feed_dict)
            import ipdb;ipdb.set_trace()
            duration = time.time() - start_time
            train_losses[step] = loss_value
            train_accs[step] = train_acc
            timesteps[step] = duration
            assert not np.isnan(
                loss_value).any(), 'Model diverged with loss = NaN'
            if step % config.validation_iters == 0:
                it_val_acc = np.asarray([])
                it_val_loss = np.asarray([])
                for num_vals in range(config.num_validation_evals):
                    # Validation accuracy as the average of n batches
                    iva, ivl = sess.run([val_accuracy, val_loss])
                    it_val_acc = np.append(it_val_acc, iva)
                    it_val_loss = np.append(it_val_loss, ivl)
                val_acc = it_val_acc.mean()
                val_loss = it_val_loss.mean()
                val_accs[step] = val_acc
                val_losses[step] = val_loss

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.batch_size / duration, float(duration),
                    train_acc, val_acc, summary_dir))

                # Save the model checkpoint if it's the best yet
                if config.top_n_validation > 0:
                    rep_idx = val_acc > val_accs
                    if sum(rep_idx) > 0:
                        force_save = True
                        val_accs[np.argmax(rep_idx)] = val_acc
                else:
                    force_save = True

                if force_save:
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        'model_' + str(step) + '.ckpt')
                    saver.save(
                        sess, ckpt_path, global_step=step)
                    print 'Saved checkpoint to: %s' % ckpt_path
                    force_save = False

                    time_elapsed += float(duration)
                    db.update_performance(
                        experiment_id=config._id,
                        experiment_name=config.experiment_name,
                        summary_dir=summary_dir,
                        ckpt_file=ckpt_path,
                        training_loss=float(loss_value),
                        validation_loss=float(val_acc),
                        time_elapsed=time_elapsed,
                        training_step=step)
                if config.early_stop:
                    keys = np.sort([int(k) for k in val_accs.keys()])
                    sorted_vals = np.asarray([val_accs[k] for k in keys])
                    if check_early_stop(sorted_vals):
                        print 'Triggered an early stop.'
                        break
            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training accuracy = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.batch_size / duration,
                                     float(duration), train_acc))

            # End iteration
            step += 1

    except:
        print 'Done training for %d epochs, %d steps.' % (1, step)
        print 'Saved to: %s' % checkpoint_dir
    sess.close()
    return train_losses, val_losses, train_accs, val_accs, timesteps
