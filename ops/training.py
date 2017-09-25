import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime


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
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        threads,
        summary_dir,
        checkpoint_dir,
        train_dict,
        val_dict,
        train_model,
        val_model,
        exp_params):
    step, time_elapsed = 0, 0
    train_losses, train_accs, timesteps = {}, {}, {}
    val_losses, val_accs, val_scores, val_labels = {}, {}, {}, {}
    try:
        while not coord.should_stop():
            start_time = time.time()
            train_vars = sess.run(train_dict.values())
            it_train_dict = {k: v for k, v in zip(
                train_dict.keys(), train_vars)}
            duration = time.time() - start_time
            train_losses[step] = it_train_dict['train_loss']
            train_accs[step] = it_train_dict['train_accuracy']
            timesteps[step] = duration
            assert not np.isnan(
                it_train_dict['train_loss']
                ).any(), 'Model diverged with loss = NaN'
            if step % config.validation_iters == 0:
                it_val_acc = np.asarray([])
                it_val_loss = np.asarray([])
                it_val_scores = np.asarray([])
                it_val_labels = np.asarray([])
                for num_vals in range(config.num_validation_evals):
                    # Validation accuracy as the average of n batches
                    val_vars = sess.run(val_dict.values())
                    it_val_dict = {k: v for k, v in zip(
                        val_dict.keys(), val_vars)}
                    it_val_acc = np.append(
                        it_val_acc,
                        it_val_dict['val_accuracy'])
                    it_val_loss = np.append(
                        it_val_loss,
                        it_val_dict['val_loss'])
                    it_val_scores = np.append(
                        it_val_loss,
                        it_val_dict['val_scores'])
                    it_val_labels = np.append(
                        it_val_loss,
                        it_val_dict['val_labels'])
                val_acc = it_val_acc.mean()
                val_lo = it_val_loss.mean()
                val_accs[step] = val_acc
                val_losses[step] = val_lo
                val_scores[step] = it_val_scores
                val_labels[step] = it_val_labels

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                print format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['train_loss'],
                    config.batch_size / duration,
                    float(duration),
                    it_train_dict['train_accuracy'],
                    val_acc,
                    summary_dir)

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
                        training_loss=float(it_train_dict['train_loss']),
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
                print format_str % (
                    datetime.now(),
                    step,
                    it_train_dict['train_loss'],
                    config.batch_size / duration,
                    float(duration),
                    it_train_dict['train_accuracy'])

            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print 'Done training for %d epochs, %d steps.' % (config.epochs, step)
        print 'Saved to: %s' % checkpoint_dir
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()

    # If using hp optimization, store performance here
    if exp_params['hp_current_iteration'] is not None:
        exp_params['hp_current_iteration'] += 1

    # Package output variables into a dictionary
    output_dict = {
        'train_losses': train_losses,
        'train_accs': val_losses,
        'timesteps': train_accs,
        'val_losses': val_accs,
        'val_accs': timesteps,
        'val_scores': val_scores,
        'val_labels': val_labels,
    }
    return output_dict
