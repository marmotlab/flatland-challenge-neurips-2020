import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

# parameters for training
NUM_CHANNELS = 22

GRAD_CLIP = 90.0

KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
KEEP_PROB3 = 1
KEEP_PROB4 = 1
KEEP_PROB5 = 0.8
KEEP_PROB6 = 0.8
RNN_SIZE = 1024
GOAL_REPR_SIZE = 12
NUM_HIDDEN_UNITS1 = 256
NUM_HIDDEN_UNITS2 = 256
NUM_HIDDEN_UNITS3 = 256
NUM_HIDDEN_UNITS4 = 256
NUM_HIDDEN_UNITS5 = 128
NUM_HIDDEN_UNITS6 = 64


# Used to initialize weights for policy and value output layers (Do we need to use that? Maybe not now)
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class ACNet:
    def __init__(self, scope, a_size, trainer, TRAINING, GLOBAL_NET_SCOPE, OBS_SIZE):
        with tf.variable_scope(str(scope) + '/qvalues'):
            self.inputs = tf.placeholder(shape=[None, OBS_SIZE], dtype=tf.float32)
            #           self.goal_pos=tf.placeholder(shape=[None,3],dtype=tf.float32)
            #           self.myinput = tf.transpose(self.inputs, perm=[0,2,3,1])
            # self.policy, self.value, self.state_out, self.state_in, self.state_init, self.valids = self._build_net(
            #     self.inputs, TRAINING, a_size, RNN_SIZE)
            self.policy, self.value, self.valids = self._build_net(
                self.inputs, TRAINING, a_size, RNN_SIZE, OBS_SIZE)

        if TRAINING:
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
            self.valid_actions = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
            self.target_v = tf.placeholder(tf.float32, [None], 'Vtarget')
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            #           self.target_collisioncourse = tf.placeholder(tf.float32, [None])
            #           self.target_astar           = tf.placeholder(shape=[None,a_size], dtype=tf.float32)
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.train_value = tf.placeholder(tf.float32, [None])
            #           self.train_astar            = tf.placeholder(tf.float32, [None])
            self.optimal_actions = tf.placeholder(tf.int32, [None])
            self.optimal_actions_onehot = tf.one_hot(self.optimal_actions, a_size, dtype=tf.float32)

            # Loss Functions
            self.value_loss = (0.005 / 4) * tf.reduce_sum(
                self.train_value * tf.square(self.target_v - tf.reshape(self.value, shape=[-1])))
            self.entropy = -0.001 * tf.reduce_sum(self.policy * tf.log(tf.clip_by_value(self.policy, 1e-10, 1.0)))
            self.policy_loss = -0.02 * tf.reduce_sum(
                tf.log(tf.clip_by_value(self.responsible_outputs, 1e-15, 1.0)) * self.advantages)
            self.valid_loss = -0.01 * tf.reduce_sum(tf.log(tf.clip_by_value(self.valids, 1e-10, 1.0)) * \
                                                    self.valid_actions + tf.log(
                tf.clip_by_value(1 - self.valids, 1e-10, 1.0)) * (1 - self.valid_actions))
            # self.collisioncourse_loss = - tf.reduce_sum(self.target_collisioncourse*tf.log(tf.clip_by_value(self.collisioncourse,1e-10,1.0))\
            #                                      +(1-self.target_collisioncourse)*tf.log(tf.clip_by_value(1-self.collisioncourse,1e-10,1.0)))
            # self.astar_loss    = - tf.reduce_sum(self.train_astar*tf.reduce_sum(tf.log(tf.clip_by_value(self.next_astar,1e-10,1.0)) *\
            #                                 self.target_astar+tf.log(tf.clip_by_value(1-self.next_astar,1e-10,1.0)) * (1-self.target_astar), axis=1))
            # self.astar_loss = tf.reduce_sum(self.train_astar*tf.contrib.keras.backend.categorical_crossentropy(self.target_astar,self.policy))
            self.loss = 1 * self.value_loss + self.policy_loss - 1 * self.entropy + 1 * self.valid_loss  # + .5*self.collisioncourse_loss +.5*self.astar_loss
            self.imitation_loss = 0.2 * tf.reduce_mean(
                tf.contrib.keras.backend.categorical_crossentropy(self.optimal_actions_onehot, self.policy))

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + '/qvalues')
            self.gradients = tf.gradients(self.loss, local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, GRAD_CLIP)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE + '/qvalues')
            self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

            # now the gradients for imitation loss
            self.i_gradients = tf.gradients(self.imitation_loss, local_vars)
            self.i_var_norms = tf.global_norm(local_vars)
            i_grads, self.i_grad_norms = tf.clip_by_global_norm(self.i_gradients, GRAD_CLIP)

            # Apply local gradients to global network
            self.apply_imitation_grads = trainer.apply_gradients(zip(i_grads, global_vars))
        #           self.homogenize_weights = update_target_graph(str(scope)+'/qvaluesB', str(scope)+'/qvalues')

        print("Hello World... From  " + str(scope))  # :)

    def _build_net(self, inputs, TRAINING, a_size, RNN_SIZE, OBS_SIZE):
        w_init = layers.variance_scaling_initializer()

        def conv_mlp(inputs, kernal_size, output_size):
            inputs = tf.reshape(inputs, [-1, 1, kernal_size, 1])
            conv = layers.conv2d(inputs=inputs, padding="VALID", num_outputs=output_size,
                                 kernel_size=[1, kernal_size], stride=1,
                                 data_format="NHWC", weights_initializer=w_init, activation_fn=tf.nn.relu)

            return conv

        conv1 = conv_mlp(inputs, OBS_SIZE, NUM_HIDDEN_UNITS1)
        conv2 = conv_mlp(conv1, NUM_HIDDEN_UNITS1, NUM_HIDDEN_UNITS2)
        conv3 = conv_mlp(conv2, NUM_HIDDEN_UNITS2, NUM_HIDDEN_UNITS3)
        conv4 = conv_mlp(conv3, NUM_HIDDEN_UNITS3, NUM_HIDDEN_UNITS4)
        conv5 = conv_mlp(conv4, NUM_HIDDEN_UNITS4, NUM_HIDDEN_UNITS5)
        conv6 = conv_mlp(conv5, NUM_HIDDEN_UNITS5, NUM_HIDDEN_UNITS6)
        self.conv_last = conv_mlp(conv6, NUM_HIDDEN_UNITS6, RNN_SIZE)
        self.conv_last = tf.reshape(self.conv_last, [-1, RNN_SIZE])

        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE, state_is_tuple=True)
        # c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        # h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        # state_init = [c_init, h_init]
        # c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        # h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        # state_in = (c_in, h_in)
        # rnn_in = tf.expand_dims(self.conv5, [0])
        # step_size = tf.shape(inputs)[:1]
        # state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
        # lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        #     lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
        #     time_major=False)
        # lstm_c, lstm_h = lstm_state
        # state_out = (lstm_c[:1, :], lstm_h[:1, :])
        # self.rnn_out = tf.reshape(lstm_outputs, [-1, RNN_SIZE])
        policy_layer = layers.fully_connected(inputs=self.conv_last, num_outputs=a_size,
                                              weights_initializer=normalized_columns_initializer(1. / float(a_size)),
                                              biases_initializer=None, activation_fn=None)

        policy = tf.nn.softmax(policy_layer)
        policy_sig = tf.sigmoid(policy_layer)
        value = layers.fully_connected(inputs=self.conv_last, num_outputs=1,
                                       weights_initializer=normalized_columns_initializer(1.0),
                                       biases_initializer=None, activation_fn=None)
        return policy, value, policy_sig
        # return policy, value, state_out, state_in, state_init, policy_sig
