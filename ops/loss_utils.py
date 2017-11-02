import tensorflow as tf
from ops.eval_metrics import pearson_score


def optimizer_interpreter(
        loss,
        lr,
        optimizer,
        model=None,
        constraints=None):
    """Router for loss functions."""
    if optimizer == 'adam':
        return tf.train.AdamOptimizer(lr).minimize(loss)
    elif optimizer == 'sgd':
        return tf.train.GradientDescentOptimizer(lr).minimize(loss)
    elif optimizer == 'momentum':
        return momentum(loss=loss, lr=lr)
    elif optimizer == 'rmsprop':
        return tf.train.RMSPropOptimizer(lr).minimize(loss)
    elif optimizer == 'hessian':
        import ipdb;ipdb.set_trace()
        var_constraints = {model[k]: v for k, v in constraints.iteritems()}
        return tf.contrib.opt.ScipyOptimizerInterface(
            loss,
            options={'maxiter': 1000},
            var_to_bounds=var_constraints,
            method='fmin_tnc')
    else:
        raise RuntimeError('Cannot understand your loss function.')


def momentum(loss, lr, momentum=0.9):
    """Wrapper for SGD with momentum."""
    tf.train.MomentumOptimizer(lr, momentum=momentum).minimize(loss)


def loss_interpreter(
        logits,
        labels,
        loss_type,
        weights=None,
        dataset_module=None):
    """Router for loss functions."""
    if loss_type is None:
        loss_type = dataset_module.default_loss_function
    if loss_type == 'cce':
        return cce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'l2':
        return l2(
            logits=logits,
            labels=labels)
    elif loss_type == 'log_loss':
        return log_loss(
            logits=logits,
            labels=labels)
    elif loss_type == 'huber':
        return huber(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'sigmoid':
        return sigmoid_ce(
            logits=logits,
            labels=labels,
            weights=weights)
    elif loss_type == 'pearson':
        return pearson_loss(
            logits=logits,
            labels=labels)
    elif loss_type == 'log_poisson':
        return log_poisson(
            logits=logits,
            labels=labels)
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
    if loss_type == 'l2' or loss_type == 'L2':
        return tf.nn.l2_loss(weight)
    elif loss_type == 'l1' or loss_type == 'L1':
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
    l2_loss = tf.nn.l2_loss(
        logits - labels)
    return l2_loss, l2_loss


def l1(logits, labels):
    """Wrapper for l2 loss."""
    l1_loss = tf.reduce_sum(
        tf.abs(logits - labels))
    return l1_loss, l1_loss


def huber(logits, labels, weights):
    """Wrapper for huber loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    if weights is None:
        weights = 1.
    return tf.losses.huber_loss(
        predictions=logits,
        labels=labels,
        weights=weights), tf.nn.l2_loss(logits - labels)


def log_loss(logits, labels):
    """Wrapper for log loss."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    ll = tf.losses.log_loss(
        predictions=logits,
        labels=labels)
    return ll, ll


def log_poisson(logits, labels, eps=1e-12):
    """Wrapper for log poisson loss from the Antilok dataset."""
    logits = tf.squeeze(logits)
    labels = tf.squeeze(labels)
    ll = tf.reduce_sum(logits) - tf.reduce_sum(
        tf.multiply(labels, tf.log(logits + eps)))
    return ll, ll


def pearson_loss(logits, labels):
    """Pearson dissimilarity loss."""
    rhos = 1 - pearson_score(pred=logits, labels=labels)
    mean_rhos = tf.reduce_mean(rhos)
    return mean_rhos, rhos


def sigmoid_ce(logits, labels, weights, force_dtype=tf.float32):
    """Wrapper for sigmoid cross entropy loss."""
    if force_dtype:
        if logits.dtype != force_dtype:
            logits = tf.cast(logits, force_dtype)
        if labels.dtype != force_dtype:
            labels = tf.cast(labels, force_dtype)
    if weights is None:
        weights = 1.
    sig_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits) * weights)
    return sig_loss, sig_loss

