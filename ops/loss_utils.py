import tensorflow as tf


def optimizer_interpreter(
        loss,
        lr,
        optimizer):
    """Router for loss functions."""
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr).minimize(loss)
    elif optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(lr).minimize(loss)
    elif optimizer == 'momentum':
        return momentum(loss=loss, lr=lr)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr).minimize(loss)
    else:
        raise RuntimeError('Cannot understand your loss function.')


def momentum(loss, lr, momentum=0.9):
    """Wrapper for SGD with momentum."""
    tf.train.MomentumOptimizer(lr, momentum=momentum).minimize(loss)


def loss_interpreter(
        logits,
        labels,
        loss_type,
        weights=None):
    """Router for loss functions."""
    if loss_type == 'cce':
        return cce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'l2':
        return l2(
            logits=logits,
            labels=labels)
    elif loss_type == 'l1':
        return l1(
            logits=logits,
            labels=labels)
    elif loss_type == 'huber':
        return huber(
            logits=logits,
            labels=labels,
            weights=weights)
    else:
        raise RuntimeError('Cannot understand your loss function.')


def wd_loss(
        regularizations,
        loss,
        wd_penalty):
    """Calculate weight decay loss and add it to the main loss."""
    regs = []
    for k, v in regularizations.iteritems():
        lt = v['regularization_type']
        ls = v['regularization_strength']
        lw = v['weight']
        regs += [interpret_reg_loss(lw, lt) * ls]
    return loss + tf.add_n(regs)


def interpret_reg_loss(weight, loss_type):
    if loss_type == 'l2':
        return tf.nn.l2_loss(weight)
    elif loss_type == 'l1':
        return tf.reduce_sum(tf.abs(weight))
    else:
        raise RuntimeError('Cannot understand regularization type.')


def cce(logits, labels, weights=None):
    """Sparse categorical cross entropy with weights."""
    if weights is not None:
        weights = tf.get_variable(
            name='weights', initializer=weights)[None, :]
        weights_per_label = tf.matmul(
            tf.one_hot(labels, 2), tf.transpose(tf.cast(weights, tf.float32)))
        return tf.reduce_mean(
            tf.multiply(
                weights_per_label,
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels))), tf.nn.softmax(logits)
    else:
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels)
            ), tf.nn.softmax(logits)


def l2(logits, labels):
    """Wrapper for l2 loss."""
    return tf.nn.l2_loss(
        logits - labels), tf.nn.l2_loss(
        logits - labels)


def l1(logits, labels):
    """Wrapper for l2 loss."""
    return tf.nn.reduce_sum(
        tf.abs(logits - labels)), tf.nn.reduce_sum(
        tf.abs(logits - labels))


def huber(logits, labels, weights):
    """Wrapper for huber loss."""
    if weights is None:
        weights = 1.
    return tf.losses.huber(
        predictions=logits,
        labels=labels,
        weights=weights), tf.nn.l2_loss(logits - labels)
