import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# ==============================Network Structure==============================#
INPUT_NODE = 84
INPUT_CHANNELS = 3
CONV_LAYER1_NODE = 8
CONV_LAYER1_DEPTH = 32
CONV_LAYER2_NODE = 4
CONV_LAYER2_DEPTH = 64
CONV_LAYER3_NODE = 3
CONV_LAYER3_DEPTH = 64
FC_LAYER1_NODE = 512
# ================================    END        ==============================#

np.random.seed(1)
tf.set_random_seed(1)


class Deep_Q_Network:
    def __init__(
            self,
            action_size,
            feature_size,
            learning_rate_base=0.00025,
            reward_decay=0.99,
            e_greedy=0.9,
            target_net_replace_round=10000,
            memory_size=20000,
            batch_size=32,
            e_greedy_increment=0.000001,
            output_graph=True,
    ):
        # number of all optional actions
        self.action_size = action_size
        # the features contained in state
        self.feature_size = feature_size
        self.feature_num = feature_size[0]*feature_size[1]*feature_size[2]
        # The degree of reliance on past experience
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        # when to replace target net parameter by eval net
        self.target_net_replace_round = target_net_replace_round
        # experience replay size
        self.memory_size = memory_size
        self.batch_size = batch_size
        # not figure out yet
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # step counter
        self.learning_step_counter = 0
        self.global_step=tf.Variable(0)
        self.saver = tf.train.Saver()
        self.learning_rate = tf.train.exponential_decay(
            learning_rate_base, self.learning_step_counter, 100, 0.99, staircase="True")
        # initialize zero memory pool [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.feature_num*2+2))
        self.ouput_graph_flag = output_graph

        self.build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        # replace params of target net by eval net params
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # create session
        self.sess = tf.Session()

        # global initialize
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # ====================== build_eval_net ===============================

        self.s = tf.placeholder(
            tf.float32, [None, self.feature_size[0], self.feature_size[1], self.feature_size[2]], name='s')
        # what if cancel this placeholder??
        self.q_target = tf.placeholder(tf.float32, [None, self.action_size], name='Q_target')
        # ======================certain net structure==========================
        with tf.variable_scope('eval_net'):
            collection_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("weight",
                                          [CONV_LAYER1_NODE, CONV_LAYER1_NODE, INPUT_CHANNELS, CONV_LAYER1_DEPTH],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("bias", [CONV_LAYER1_DEPTH], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(self.s, weights, strides=[1, 4, 4, 1], padding='VALID')
                relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
            with tf.variable_scope('conv2'):
                weights = tf.get_variable("weight",
                                          [CONV_LAYER2_NODE, CONV_LAYER2_NODE, CONV_LAYER1_DEPTH, CONV_LAYER2_DEPTH],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("bias", [CONV_LAYER2_DEPTH], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(relu1, weights, strides=[1, 2, 2, 1], padding='VALID')
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
            with tf.variable_scope('conv3'):
                weights = tf.get_variable("weight",
                                          [CONV_LAYER3_NODE, CONV_LAYER3_NODE, CONV_LAYER2_DEPTH, CONV_LAYER3_DEPTH],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("bias", [CONV_LAYER3_DEPTH], initializer=tf.constant_initializer(0.0))
                conv3 = tf.nn.conv2d(relu2, weights, strides=[1, 1, 1, 1], padding='VALID')
                relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases))

                conv_out_shape = relu3.get_shape().as_list()
                print(conv_out_shape)
                nodes = conv_out_shape[1]*conv_out_shape[2]*conv_out_shape[3]
                print(nodes)
                reshaped = flatten(relu3)

            with tf.variable_scope('fc1'):
                weights = tf.get_variable("weights",
                                          [nodes, FC_LAYER1_NODE],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("biases",
                                         [FC_LAYER1_NODE],
                                         initializer=tf.constant_initializer(0.1),
                                         collections=collection_name)
                fc1 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)

            with tf.variable_scope('output'):
                weights = tf.get_variable("weights",
                                          [FC_LAYER1_NODE, self.action_size],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("biases",
                                         [self.action_size],
                                         initializer=tf.constant_initializer(0.1),
                                         collections=collection_name)
                self.q_eval = tf.matmul(fc1, weights + biases)

        # =====================================================================
        self.s_ = tf.placeholder(
            tf.float32, [None, self.feature_size[0], self.feature_size[1], self.feature_size[2]], name='s_')
        # ====================== build_target_net =============================
        with tf.variable_scope('target_net'):
            collection_name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("weight",
                                          [CONV_LAYER1_NODE, CONV_LAYER1_NODE, INPUT_CHANNELS, CONV_LAYER1_DEPTH],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("bias", [CONV_LAYER1_DEPTH], initializer=tf.constant_initializer(0.0))
                conv1 = tf.nn.conv2d(self.s, weights, strides=[1, 4, 4, 1], padding='VALID')
                relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
            with tf.variable_scope('conv2'):
                weights = tf.get_variable("weight",
                                          [CONV_LAYER2_NODE, CONV_LAYER2_NODE, CONV_LAYER1_DEPTH, CONV_LAYER2_DEPTH],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("bias", [CONV_LAYER2_DEPTH], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(relu1, weights, strides=[1, 2, 2, 1], padding='VALID')
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
            with tf.variable_scope('conv3'):
                weights = tf.get_variable("weight",
                                          [CONV_LAYER3_NODE, CONV_LAYER3_NODE, CONV_LAYER2_DEPTH, CONV_LAYER3_DEPTH],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("bias", [CONV_LAYER3_DEPTH], initializer=tf.constant_initializer(0.0))
                conv3 = tf.nn.conv2d(relu2, weights, strides=[1, 1, 1, 1], padding='VALID')
                relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases))

                conv_out_shape = relu3.get_shape().as_list()
                nodes = conv_out_shape[1]*conv_out_shape[2]*conv_out_shape[3]
                reshaped = flatten(relu3)

            with tf.variable_scope('fc1'):
                weights = tf.get_variable("weights",
                                          [nodes, FC_LAYER1_NODE],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("biases",
                                         [FC_LAYER1_NODE],
                                         initializer=tf.constant_initializer(0.1),
                                         collections=collection_name)
                fc1 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)

            with tf.variable_scope('output'):
                weights = tf.get_variable("weights",
                                          [FC_LAYER1_NODE, self.action_size],
                                          initializer=tf.random_normal_initializer(0., 0.3),
                                          collections=collection_name)
                biases = tf.get_variable("biases",
                                         [self.action_size],
                                         initializer=tf.constant_initializer(0.1),
                                         collections=collection_name)
                self.q_next = tf.matmul(fc1, weights + biases)
        # =====================================================================
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def store_in_memory(self, s, a, r, s_):
        # if self dont have member memory counter:
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # put state into 1 dimension so that it can be saved in memory
        s = np.array(s).flatten()
        s_ = np.array(s_).flatten()
        # print(s)
        transition = np.hstack((s, [a, r], s_))
        # fill the memory pool
        index = self.memory_counter % self.memory_size
        # [x,:] means all member in line x
        self.memory[index, :] = transition
        self.memory_counter += 1

    def read_memory(self, batch_memory):
        s_memory = np.zeros((self.batch_size, self.feature_size[0], self.feature_size[1], self.feature_size[2]))
        s__memory = np.zeros((self.batch_size, self.feature_size[0], self.feature_size[1], self.feature_size[2]))
        s_temp = batch_memory[:, :self.feature_num]
        s__temp = batch_memory[:, -self.feature_num:]
        for index in range(batch_memory.shape[0]):
            s_memory[index, :] = s_temp[index].reshape((84, 84, 3))
            s__memory[index, :] = s__temp[index].reshape((84, 84, 3))
            action = batch_memory[index, self.feature_num]
            reward = batch_memory[index, self.feature_num+1]
        return s_memory, s__memory, action, reward

    def choose_action(self, observation):
        # change the observation dimension into [1,feature_num]
        observation = observation[np.newaxis, :]
        # in 90% possibility
        if np.random.uniform() < self.epsilon:
            # action_value is a list
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # action is a num
            action = np.argmax(actions_value)
        # in 10% possibility
        else:
            action = np.random.randint(0, self.action_size)
        return action

    def learn(self):
        # check to replace target parameters or not
        if self.learning_step_counter % self.target_net_replace_round == 0:
            self.sess.run(self.replace_target_op)
            print("params replaced")

        # get sample batch memory from memory pool
        # if pool is replete
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # choose the sample index line to constitute batch
        batch_memory = self.memory[sample_index, :]

        s_input, s__input, eval_act, reward = self.read_memory(batch_memory)

        # inference two network
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval], feed_dict={
                                                self.s_: s__input,
                                                self.s: s_input
                                                }
        )

        # method to get q_target
        q_target = q_eval.copy()
        # pick out the samples which are used to train, return a list
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # return signal int present the index of action
        eval_act_index = eval_act.astype(int)

        # following sentence is hard to understand for me at first
        # type of q_target is a [32,6] mat, first dimension is batch size, second is [s,a,r,s_]
        # following operation just like pick up the value of action executed in each line and make
        # them up to a list, the member of this list point to the origin member in mat q_target
        # just like a pointer list!!
        # in debug what I got was a list with 32 member present value of action executed in each line
        # what a fucking sao cao zuo
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        q_loss = q_target-q_eval   # just for watch
        # train the eval network
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={
                                                self.s: s_input,
                                                self.q_target: q_target
                                        }
                                    )
        # increasing the epsilon so that with the training step increasing,
        # the possibility of taking random action will decrease
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learning_step_counter += 1

        if self.learning_step_counter % 50 == 0:
            print("after", self.learning_step_counter,
                  "step, the losses is ", self.cost,
                  "epsilon is", self.epsilon,
                  "learning_rate is", self.sess.run(self.learning_rate))

        if self.ouput_graph_flag and self.learning_step_counter % 5000 == 0:
            self.saver.save(self.sess, '/CNN_ATARI_MODEL/model.ckpt')
            print("model saved")
