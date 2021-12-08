# Train the averModel for initializing LISTA_CEHyper.
import sys
import tensorflow as tf
import scipy.io as scio
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as pyplot
import argparse
import random
from numpy import linalg as la
from tensorflow.python.client import timeline

parser = argparse.ArgumentParser(description='parameters to LISTA_CE')
parser.add_argument('--CS_ratio', type=float, default=2)
parser.add_argument('--layers', type=int, default=20)
parser.add_argument('--Train_Flag', type=int, default=1)
# args = parser.parse_args(args=[])
args = parser.parse_args()

# %matplotlib inline
reuse = tf.AUTO_REUSE
fdtype = np.float32
tf.reset_default_graph()

# parameter for wideband OFDM - MIMO system
CS_ratio = args.CS_ratio
CS_ratio_tag = str(int(CS_ratio * 100)).zfill(3)
complex_label = 2  # 2 for complex signal, 1 for real signal
Nc = 32  # number of subcarriers
Nt = 32  # number of transmitted antennas
M = int(complex_label * Nc * Nt * CS_ratio)

# parameter for LISTA_CE
max_iteration = args.layers

# parameter for Training
lr = 0.0001
train_batchsize = 64
test_batchsize = 1024
aver_epoch = 1000
max_episode = 1000
Test_epoch = max_episode

Train_flag = args.Train_Flag  # 1 for Training, 0 for Testing
Draw_flag = 0   # 1 for Draw Picture, 0 for No Draw Picture
Layer_by_layer_flag = 0  # 1 for layer_by_layer, 0 for otherwise
timeline_flag = 0 # 1 for Draw timeline

if Train_flag == 1:
    L_set = [2, 3, 4]
    SNR_range = [5, 15]
    SNR_interval = 2.5
    SNR_sample_len = (SNR_range[1] - SNR_range[0]) // SNR_interval + 1
    SNR_sample = np.linspace(SNR_range[0], SNR_range[1], SNR_sample_len, dtype=np.float32)
else:
    L_set = [1, 2, 3, 4, 5, 6]
    SNR_range = [0, 20]
    SNR_interval = 2.5
    SNR_sample_len = (SNR_range[1] - SNR_range[0]) // SNR_interval + 1
    SNR_sample = np.linspace(SNR_range[0], SNR_range[1], SNR_sample_len, dtype=np.float32)


appendix_tag_averModel = "_averModel"
type_tag_aver = "CS" + CS_ratio_tag + "_layer" + str(max_iteration)
model_dir_aver = 'LISTA_CE_' + type_tag_aver + appendix_tag_averModel
output_file_name = "Log_output_%s.txt" % model_dir_aver


def Log_out(Log_out_string, Log_dir=output_file_name):
    print(Log_out_string)
    output_file = open(output_file_name, 'a')
    output_file.write(Log_out_string)
    output_file.write('\n')
    output_file.close()


def write_parameters_log(output_file_name, train_flag):
    startTime = datetime.now()
    Log_out("Running begin at " + str(startTime))
    Log_out("Train_flag = " + str(Train_flag))
    if Train_flag == 1:
        Log_out("### parameter for wideband OFDM-MIMO system ###")
        Log_out("CS_ratio = " + str(CS_ratio))
        Log_out("Num of subcarriers, Nc = " + str(Nc))
        Log_out("Num of antennas, Nt = " + str(Nt))
        Log_out("\n ### parameter for LISTA_CE ###")
        Log_out("Layer of LISTA_CE = " + str(max_iteration))
        Log_out("\n ### parameter for Training ###")
        Log_out("Learning rate = " + str(lr))
        Log_out("model_dir_aver = " + str(model_dir_aver))


def phi_gen():  # Generate \bar{W} in equ.(8)
    block_h = Nt
    block_w = int(Nt * CS_ratio)
    block = 1. / np.sqrt(block_w) * (2 * np.random.randint(0, 2, size=(block_h, block_w)) - 1)  # 64*32
    Phi_input = np.zeros([block_h * Nc * complex_label, block_w * Nc * complex_label], dtype='float32')  # 4096*2048
    for index in range(Nc * complex_label):
        Phi_input[index * block_h: (index + 1) * block_h, index * block_w: (index + 1) * block_w] = block
    return Phi_input


def load_dataset(L_set):
    Train_data = []
    Vali_data = []
    Test_data = []
    for Path_num in L_set:
        dataset_dir = '_data_LISTA_CE_BeamFreq_Path' + str(Path_num) + '.mat'
        Log_out("Load dataset from Train" + dataset_dir)
        mat = scio.loadmat('Train' + dataset_dir)
        Train_data.append(mat['train_data'])  # array
        mat = scio.loadmat('Vali' + dataset_dir)
        Vali_data.append(mat['vali_data'])  # array
        mat = scio.loadmat('Test' + dataset_dir)
        Test_data.append(mat['test_data'])  # array

    Train_data = np.array(Train_data)
    Vali_data = np.array(Vali_data)
    Test_data = np.array(Test_data)
    Train_data = np.reshape(Train_data, [Train_data.shape[0] * Train_data.shape[1], Train_data.shape[2]])
    Log_out("Train_data.shape = " + str(Train_data.shape))
    Log_out("Vali_data.shape = " + str(Vali_data.shape))
    Log_out("Test_data.shape = " + str(Test_data.shape))
    return Train_data, Vali_data, Test_data


Phi_input = phi_gen()
write_parameters_log(output_file_name, Train_flag)
Training_inputs, Vali_inputs, Test_inputs = load_dataset(L_set)


def variable_w(shape):
    w = tf.get_variable('w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1))  # 此处被修改了
    return w


def variable_b(shape, initial=0.01):
    b = tf.get_variable('b', shape=shape, initializer=tf.constant_initializer(initial))
    return b


def function_g(x, Phi_tf): # Multiply channel matrix with \bar{W}.
    x_compress = tf.matmul(x, Phi_tf)
    return x_compress


def function_fs(x, layer_num):  # Transformation, F, in equ.(12)
    '''
    complex_label * Nt --> 256
    '''
    input_dim = complex_label * Nt
    hidden_dim = 128
    output_dim = 256
    if layer_num % 2 == 0:  # Trans in equ.(16)
        x = tf.reshape(x, [-1, Nc, input_dim])
        x_left= x[:, :, 0:Nt]
        x_right = x[:, :, Nt:]
        x_left = tf.transpose(x_left, perm=[0, 2, 1])
        x_right = tf.transpose(x_right, perm=[0, 2, 1])
        x = tf.concat([x_left, x_right], 2)
    x = tf.reshape(x, [-1, input_dim])
    with tf.variable_scope('layer_%d/fs.1' % layer_num, reuse=reuse):
        w = variable_w([input_dim, hidden_dim])
        b = variable_b([hidden_dim])
        l = tf.nn.relu(tf.matmul(x, w) + b)
    with tf.variable_scope('layer_%d/fs.2' % layer_num, reuse=reuse):
        w = variable_w([hidden_dim, output_dim])
        b = variable_b([output_dim])
        l_out = tf.matmul(l, w) + b
    return l_out


def function_fd(x, layer_num):  # Invert transformation, \tilde{F}, in equ.(12)
    '''
    256 --> complex_label * Nt
    '''
    input_dim = 256
    hidden_dim = 128
    output_dim = complex_label * Nt
    with tf.variable_scope('layer_%d/fd.1' % layer_num, reuse=reuse):
        w = variable_w([input_dim, hidden_dim])
        b = variable_b([hidden_dim])
        l = tf.nn.relu(tf.matmul(x, w) + b)
    with tf.variable_scope('layer_%d/fd.2' % layer_num, reuse=reuse):
        w = variable_w([hidden_dim, output_dim])
        b = variable_b([output_dim])
        l = tf.matmul(l, w) + b
    if layer_num % 2 == 0:  # # Trans' in equ.(16)
        l = tf.reshape(l, [-1, Nc, output_dim])
        x_left= l[:, :, 0:Nt]
        x_right = l[:, :, Nt:]
        x_left = tf.transpose(x_left, perm=[0, 2, 1])
        x_right = tf.transpose(x_right, perm=[0, 2, 1])
        l = tf.concat([x_left, x_right], 2)
    x_recon = tf.reshape(l, [-1, Nc * output_dim])
    return x_recon


def function_soft(x, threshold):  # soft denoiser in equ.(13)
    return tf.sign(x) * tf.maximum(0., tf.abs(x) - threshold)


def ista_block(Hv, layer_num, W, s, Phi_tf):  # Construct half layer of LISTA_CE
    with tf.variable_scope('layer_%d' % layer_num, reuse=reuse):
        rho = tf.Variable(0.15, dtype=fdtype, name='rho')
        theta = tf.Variable(0.15, dtype=fdtype, name='theta')
    r = Hv + rho * tf.matmul(s - function_g(Hv, Phi_tf), W)
    Hv_k = function_fd(function_soft(function_fs(r, layer_num), theta), layer_num)
    Hv_k_output = r + Hv_k
    return Hv_k_output


def inference_ista():  # LISTA_CE
    Hv_hat = []
    Hv0 = tf.zeros(tf.shape(Hv_init), dtype=fdtype)
    Hv_hat.append(Hv0)
    W = tf.transpose(Phi_tf)
    s = function_g(Hv_init, Phi_tf)
    sigma_w = tf.sqrt(1 / tf.pow(10., SNR / 10.))
    s = s + tf.random_normal(shape=tf.shape(s), dtype=fdtype) * sigma_w
    for i in range(max_iteration):
        Hv_hat_k = ista_block(Hv_hat[-1], i, W, s, Phi_tf)
        Hv_hat.append(Hv_hat_k)
    return Hv_hat


def compute_cost(Hv_hat, Hv_init):
    cost = []
    cost_rec = 0
    for n_layer in range(max_iteration):
        cost_rec = cost_rec + tf.reduce_mean(tf.square(Hv_hat[n_layer + 1] - Hv_init))
        cost.append(cost_rec)
    return cost


def run_vali(sess, run_type="Vali", snr_in=10, l_in=3, No_random_flag=0):
    if No_random_flag != 1:
        snr_in = SNR_sample[random.randint(0, len(SNR_sample) - 1)]
        l_in = L_set[random.randint(0, len(L_set) - 1)]
    if Train_flag == 0:
        print("Run ", run_type, ", l = ", l_in, " SNR = ", snr_in)

    if run_type == "Vali":
        rand_inds = np.random.choice(Vali_inputs.shape[1], test_batchsize, replace=False)
        batch_xs = Vali_inputs[l_in - L_set[0]][rand_inds][:]
    elif run_type == "Test":
        rand_inds = np.random.choice(Test_inputs.shape[1], test_batchsize, replace=False)
        batch_xs = Test_inputs[l_in - L_set[0]][rand_inds][:]
    else:
        print("Type error!!!")
        os._exit()

    if Train_flag == 0 and Draw_flag == 1:
        print("Draw validation inputs picture")
        input_reshape = np.reshape(batch_xs[0, :], [Nc, 2 * Nt])
        pyplot.imshow(input_reshape)
        pyplot.show()

    # # draw Hv_out of all layers
    Hv_out = sess.run([Hv_hat], feed_dict={Hv_init: batch_xs, Phi_tf: Phi_input, SNR: [snr_in]})
    Hv_out = np.asarray(Hv_out)
    loss_by_layers_NMSE = np.zeros((1, Hv_out.shape[1]))
    for ii in range(Hv_out.shape[1]):
        Hv_test_all = Hv_out[0, ii, :, :]
        for jj in range(test_batchsize):
            loss_by_layers_NMSE[0, ii] = loss_by_layers_NMSE[0, ii] + np.square(np.linalg.norm((Hv_test_all[jj, :] - batch_xs[jj, :]), ord=2)) / np.square(np.linalg.norm(batch_xs[jj, :], ord=2))  # NMSE
        loss_by_layers_NMSE[0, ii] = loss_by_layers_NMSE[0, ii] / test_batchsize

    if Train_flag == 0 and Draw_flag == 1:
        print("Draw curve of NMSE by layers")
        x1 = np.linspace(1, loss_by_layers_NMSE.shape[1] - 1, loss_by_layers_NMSE.shape[1] - 1)
        pyplot.semilogy(x1, loss_by_layers_NMSE[0, 1:])
        pyplot.xlabel('layers')
        pyplot.ylabel('NMSE')
        pyplot.show()
        print("loss_by_layers_NMSE = ", loss_by_layers_NMSE)
    return loss_by_layers_NMSE


def run_vali_AllPoint(sess, run_type="Vali"):
    NMSE_array = np.zeros([len(L_set), len(SNR_sample)])
    for i in range(len(L_set)):
        for j in range(len(SNR_sample)):
            NMSE_list = run_vali(sess, run_type, SNR_sample[j], L_set[i], 1)
            NMSE = NMSE_list[-1][-1]
            NMSE_array[i, j] = NMSE
    if Train_flag == 1:
        Log_out("NSME_array = " + str(NMSE_array.shape) + str(NMSE_array))
    else:
        print("NSME_array = ", NMSE_array.shape, NMSE_array)
        np.savetxt("Array_" + model_dir_aver + ".csv", NMSE_array, delimiter=",")


if __name__ == "__main__":
    tf.reset_default_graph()
    with tf.Graph().as_default():
        Hv_init = tf.placeholder(dtype=fdtype, shape=[None, complex_label * Nc * Nt])
        Phi_tf = tf.placeholder(dtype=fdtype, shape=[complex_label * Nc * Nt, complex_label * Nc * Nt * CS_ratio])
        SNR = tf.placeholder(dtype=fdtype, shape=[1, ])

        Hv_hat = inference_ista()
        cost_all = compute_cost(Hv_hat, Hv_init)

        with tf.variable_scope('opt', reuse=reuse):  # This is left for layer_by_layer training
            n_layer = max_iteration - 1
            opt = (tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_all[n_layer],
                                                                                  var_list=tf.trainable_variables()))

        dict_LISTA_CE = {}
        dict_Hyper = {}

        for v in tf.trainable_variables():
            if "Hyper" not in v.name:
                dict_LISTA_CE[v.name] = v
        model_LISTA_CE = tf.train.Saver(dict_LISTA_CE)
        model_All = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            if Train_flag == 0:
                LISTA_CE_save_dir = model_dir_aver + '/ReSaved_Model_LISTA_CE_epoch' + str(max_episode)
                save_dir = model_dir_aver + '/Saved_Model_LISTA_CE_epoch' + str(max_episode)
                model_All.restore(sess, save_dir)
                model_LISTA_CE.save(sess, LISTA_CE_save_dir)
                model_LISTA_CE.restore(sess, LISTA_CE_save_dir)
                run_vali_AllPoint(sess)  
            else:
                loss_episode = list()
                run_vali_AllPoint(sess)
                Train_startTime = datetime.now()
                print("Begin training……")
                for epoch_i in range(max_episode + 1):
                    epoch_startTime = datetime.now()
                    loss_batch = list()
                    rand_inds = np.random.choice(Training_inputs.shape[0], Training_inputs.shape[0], replace=False)
                    for i in range(Training_inputs.shape[0] // train_batchsize):
                        Phi_input = phi_gen()
                        snr = np.random.uniform(SNR_range[0],SNR_range[1])
                        batch_xs = Training_inputs[rand_inds[i * train_batchsize:(i + 1) * train_batchsize]][:]
                        _, cost_ = sess.run([opt, cost_all[max_iteration - 1]], feed_dict={Hv_init: batch_xs, Phi_tf: Phi_input, SNR:[snr]})
                        loss_batch.append(cost_)
                    loss_episode.append(np.mean(loss_batch))
                    nowTime = datetime.now()
                    Train_diffTime = nowTime - Train_startTime
                    epoch_diffTime = nowTime - epoch_startTime
                    if epoch_i != 0:
                        restTime = Train_diffTime / epoch_i * (max_episode - epoch_i)
                    else:
                        restTime = Train_diffTime
                    endTime = nowTime + restTime
                    epoch_time = str(epoch_diffTime.seconds) + '.' + str(epoch_diffTime.microseconds)
                    if not os.path.exists(model_dir_aver):
                        os.makedirs(model_dir_aver)
                    if epoch_i <= 1:
                        save_dir = model_dir_aver + '/Saved_Model_LISTA_CE_epoch' + str(epoch_i)
                        model_All.save(sess, save_dir, write_meta_graph=False)
                    else:
                        if epoch_i % 500 == 0:
                            save_dir = model_dir_aver + '/Saved_Model_LISTA_CE_epoch' + str(epoch_i)
                            model_All.save(sess, save_dir, write_meta_graph=False)
                        if epoch_i % 10 == 0:
                            output_data = "layer:[%d/%d] epoch:[%d/%d] cost: %.5f, cost_time: %.2f, may end at: " % (
                            max_iteration, max_iteration, epoch_i, max_episode, loss_batch[-1],
                            float(epoch_time)) + str(endTime) + '\n'
                            Log_out(output_data)
                        if epoch_i % 200 == 0:
                            run_vali_AllPoint(sess)                
                LISTA_CE_save_dir = model_dir_aver + '/ReSaved_Model_LISTA_CE_epoch' + str(max_episode)
                model_LISTA_CE.save(sess, LISTA_CE_save_dir)
                run_vali_AllPoint(sess, "Test")
