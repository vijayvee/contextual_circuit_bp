import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
import numpy as np

@tf.RegisterGradient("SymmetricConv")
def _Conv2DGrad(op, grad):
    strides = op.get_attr("strides")
    padding = op.get_attr("padding")
    use_cudnn_on_gpu = op.get_attr("use_cudnn_on_gpu")
    data_format = op.get_attr("data_format")
    shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
    dx = nn_ops.conv2d_backprop_input(
           shape_0,
           op.inputs[1],
           grad,
           strides=strides,
           padding=padding,
           use_cudnn_on_gpu=use_cudnn_on_gpu,
           data_format=data_format)
    dw = nn_ops.conv2d_backprop_filter(
           op.inputs[0],
           shape_1,
           grad,
           strides=strides,
           padding=padding,
           use_cudnn_on_gpu=use_cudnn_on_gpu,
           data_format=data_format)
    dw_t = tf.transpose(dw,(2,3,0,1))
    dw_symm_t = (0.5) * (dw_t + tf.transpose(dw_t,(1,0,2,3)))
    dw_symm = tf.transpose(dw_symm_t,(2,3,0,1))
    return dx,dw_symm

def get_conv_var(n_in, n_out, name, symmetry=True):
    if symmetry:
        conv_w_ = tf.Variable(name=name,initial_value=np.random.random((3,3,n_in,n_out)),dtype=tf.float32)
        conv_w_t = tf.transpose(conv_w_, (2,3,0,1))
        conv_w_symm = (0.5)*(conv_w_t + tf.transpose(conv_w_t,(1,0,2,3)))
        conv_w = tf.transpose(conv_w_symm, (2,3,0,1))
    else:
        conv_w = tf.Variable(name=name,initial_value=np.random.random((3,3,n_in,n_out)),dtype=tf.float32)
    return conv_w

def build_model(inp_ph, gt_ph):
    g = tf.get_default_graph()
    conv1_w = get_conv_var(n_in=10, n_out=10, name='conv1',symmetry=True)
    with g.gradient_override_map({"Conv2D": "SymmetricConv"}):
        conv1 = tf.nn.conv2d(inp_ph, conv1_w, strides=[1,1,1,1], padding='SAME')
    conv2_w = get_conv_var(n_in=10, n_out=10, name='conv2',symmetry=True)
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1,1,1,1], padding='SAME')
    loss = tf.reduce_mean(gt_ph - conv2)
    optim = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    grad_w2 = tf.gradients(loss, conv2_w)[0]
    grad_w1 = tf.gradients(loss, conv1_w)[0]
    return {'conv1_w':conv1_w, 'conv2_w':conv2_w}, {'conv1':conv1, 'conv2':conv2},{'grad_w1':grad_w1, 'grad_w2':grad_w2},loss, optim

def main():
    inp_ph = tf.placeholder(tf.float32, shape=[20,20,10,10])
    gt_ph = tf.placeholder(tf.float32, shape=[20,20,10,10])
    weights, convs, grads, loss, step = build_model(inp_ph, gt_ph)
    inp_rand = np.random.random((20,20,10,10))
    gt_rand = np.random.random((20,20,10,10))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ii in range(0,10):
            conv1_w, conv2_w = sess.run([weights['conv1_w'], weights['conv2_w']])
            grad_w1, grad_w2, loss_, _ = sess.run([grads['grad_w1'],grads['grad_w2'],loss,step],feed_dict={inp_ph:inp_rand, gt_ph:gt_rand})
            print "Iter: %s, Loss: %s"%(ii+1, loss_)
            for i in range(10):
                for j in range(10):
                    if i<j:
                        g1 = (grad_w1[:,:,i,j]==grad_w1[:,:,j,i]).all()
                        g2 = (grad_w2[:,:,i,j]==grad_w2[:,:,j,i]).all()
                        w1 = (conv1_w[:,:,i,j]==conv1_w[:,:,j,i]).all()
                        w2 = (conv2_w[:,:,i,j]==conv2_w[:,:,j,i]).all()
                        print "i:%s,j:%s,Grad_Conv1:%s,Grad_Conv2:%s,Weights_Conv1:%s,Weights_Conv2:%s"%(i,j,g1,g2,w1,w2)

if __name__=="__main__":
    main()
