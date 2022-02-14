"""
This is the code associated to the paper "Low Dimensional State Representation Learning with Reward-shaped Priors"
Botteghi N., et al, published in the International Conference of Pattern Recognition (ICPR), January 2021.

The StateRepresentation class includes the encoder neural network for learning a low-dimensional state representation
from high-dimesional observations (lidar data points + RGB camera images). The encoder is trained with a new set of
robotics priors shaping the state representation using the reward function.

This code implementation also includes the set of original priors from Rico Jonschkowski et al. (2015) extended to continuous
action spaces, i.e. the equality between two action is replaced by the similarity in magnitude. The learn() function uses
the flag use_action to select between the action-based priors and the reward-priors.
"""

import tensorflow as tf
# Hide some depreacation warnings and disable eager execution
tf.logging.set_verbosity(tf.logging.ERROR)

import os
import numpy as np
import sonnet as snt
from utils import load_pickle, reshape_observation, save_pickle
from tqdm import tqdm

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.sample_nr_buf = np.zeros(int(size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

    def store(self, obs, act, rew, done, sample_nr):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.sample_nr_buf[self.ptr] = sample_nr
        self.ptr = (self.ptr+1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size, eps=0.05, reps=0.5, use_acts=False):
        """ sample a batch which contains samples which have similar actions/rewards to the provided batch
        Current implementation: No guarantee that all samples will be visited. Possible that a sample will be
        matched with itself

         The hyperparameters eps and reps need tuning when changing environment and/or reward function. In our work,
         we used grid-search, but more advance techniques might be used instead.
         """
        splits = self.size // 1000

        idxs = np.random.randint(0, self.size-2, size=batch_size) # sample batch from memory
        idxs = idxs[self.done_buf[idxs] != 1.] # remove samples which are at the end of an episode

        a_idxs, r_idxs = [], []
        rew_delt_batch = self.rews_buf[idxs] - self.rews_buf[idxs+1]
        for j, act in enumerate(self.acts_buf[idxs]):
            sp = np.random.randint(0, splits)
            for i_s in range(splits):
                s = sp%splits

                if use_acts:
                    acts_split = self.acts_buf[s*(self.size//splits) : min((s+1) * (self.size//splits), self.size-2)]
                    # randomly draw a sample from memory
                    d = np.linalg.norm(acts_split-act, ord=2, axis=1)
                    a_ind = np.where(d < eps)[0]
                    r = np.linalg.norm(self.rews_buf[a_ind].reshape(-1,1)-self.rews_buf[idxs][j], ord=1, axis=1)
                    r_ind = a_ind[np.where(r > reps)[0]]
                else:
                    acts_split = self.acts_buf[s*(self.size//splits) : min((s+1) * (self.size//splits), self.size-2)]
                    # randomly draw a sample from memory
                    d = np.linalg.norm(acts_split-act, ord=2, axis=1)
                    a_ind = np.where(d < eps)[0]
                    r = np.linalg.norm(self.rews_buf[a_ind].reshape(-1,1)-self.rews_buf[idxs][j], ord=1, axis=1)
                    r_ind = a_ind[np.where(r > reps)[0]]

                    # use delta reward
                    rews_split = self.rews_buf[s*(self.size//splits) : min((s+1) * (self.size//splits), self.size-4) ] - \
                                 self.rews_buf[s*(self.size//splits)+1 : min((s+1) * (self.size//splits)+1, self.size-3)]
                    a_ind = np.where(abs(rews_split-rew_delt_batch[j]) < eps)[0]

                if len(r_ind) > 0 or i_s+1 == splits:
                    break
                sp+=1

            if len(a_ind) > 0:
                a_i = (sp%splits)*(self.size//splits) + np.random.choice(a_ind)
            else:
                a_i = np.random.randint(0,self.size-3)
            a_idxs.append(a_i)
            if len(r_ind) > 0:
                r_i = (sp%splits)*(self.size//splits) + np.random.choice(r_ind)
            else:
                r_i = np.random.randint(0,self.size-3)
            r_idxs.append(r_i)   

        a_idxs = np.array(a_idxs)
        r_idxs = np.array(r_idxs)

        return dict(obs=self.obs_buf[idxs], acts=self.acts_buf[idxs]), dict(obs=self.obs_buf[idxs+1]),\
               dict(obs=self.obs_buf[a_idxs]), dict(obs=self.obs_buf[a_idxs+1]), dict(obs=self.obs_buf[r_idxs])


    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size], 
                    sample_nr=self.sample_nr_buf[:self.size])

    def remove_all(self):
        self.__init__(self.obs_dim, self.act_dim, self.size)

class StateRepresentation(object):
    """ This class takes care of learning the state representation """

    def __init__(self, obs_dim, state_dim=5, batch_size=256, learning_rate=0.001, seed=1, continuelearning=False, usingros=False):

        folder = "training_results"
        filename = 'SRLnetwork'
        if usingros:
            import rospkg
            self.model_path = os.path.join(rospkg.RosPack().get_path("rosbot_srl"), folder, "srl", filename + ".ckpt")
        else:
            self.model_path = os.path.join("..", folder, "srl", filename + ".ckpt")

        self.obs_dim = obs_dim
        self.obs_count = obs_dim[0]*obs_dim[1]*3 + obs_dim[2]
        self.continuelearning = continuelearning

        self.obs_dim = obs_dim   # size [32, 24, 40] with 32x24x3 --> size RGB image and 40 --> size of lidar array
        self.obs_count = obs_dim[0] * obs_dim[1] * 3 + obs_dim[2]  #  calculate the flattened dimension of the observation vector
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.continuelearning = continuelearning

        self.memory = ReplayBuffer(self.obs_count, act_dim=2, size=30000)

        tf.random.set_random_seed(seed=seed)
        np.random.seed(seed=seed)

        # DEFINE THE FUNCTIONS REQUIRED FOR TRAINING
        self.graph = tf.Graph()
        with self.graph.as_default():
        # Define the placeholders
            self.obs_1 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_1')
            self.obs_2 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_2')
            self.obs_3 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_3')
            self.obs_4 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_4')
            self.obs_5 = tf.placeholder(tf.float32, shape=[None, self.obs_count], name='observation_5')

            self.act_1 = tf.placeholder(tf.float32, shape=[None, 2], name='action_1')

            self.is_training = tf.placeholder(tf.bool, shape=[], name="train_cond")
            
            # Define the neural network architecture and its output
            self.nn = snt.Module(self.SRLencoder, name='SRL_Network')
            self.s1 = self.nn(self.obs_1, self.is_training)
            self.s2 = self.nn(self.obs_2, self.is_training)

            self.state_delta = self.s1 - self.s2
            self.s3 = self.nn(self.obs_3, self.is_training)
            self.s4 = self.nn(self.obs_4, self.is_training)
            self.state_delta_2 = self.s3 - self.s4
            self.s5 = self.nn(self.obs_5, self.is_training)

            # define losses (i.e. the robotics priors) and optimizer (ADAM)

            self.temp_coh_loss = self.temporal_coherence_prior(self.state_delta)

            self.prop_loss = self.proportionality_prior(self.state_delta, self.state_delta_2)
            
            self.repeat_loss = self.repeatability_prior(self.s1, self.s3, self.state_delta, self.state_delta_2)

            self.caus_loss = self.causality_prior(self.s1, self.s5)

            graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_regularization_loss = tf.reduce_sum(graph_regularizers)

            self.losses = [0, 1*self.temp_coh_loss, 5*self.caus_loss, 5*self.prop_loss, 5*self.repeat_loss, 1*total_regularization_loss]
            self.loss = tf.reduce_sum(self.losses)
            self.losses[0] = tf.reduce_sum(self.losses[:-1])

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss) 

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

        # Initialize the session
        self.sess = tf.Session(graph=self.graph)
        # Restore session if desired
        if continuelearning:
            print('Loading SR model from memory')
            self.load_model()
        else:
            print("Starting SR model from scratch!")
            self.sess.run(self.init)

    def SRLencoder(self, observations, is_training, l2_reg=0.001, batch_norm=False):

        """
        The camera image and the lidar data points are separated from the flattened observation vector via snt.SliceByDim()
        in camera_inputs and laser_inputs. camera_inputs are reshaped into [32, 24, 3] and fed to conv2d layers,
        while laser_inputs are reshaped into [40, 1] and fed to a conv1d layer. Noise is added to the output during training
        for preventing equal state predictions (for more details we reference to "Learning State Representations with Robotics Priors"
        Jonschkowski R. et al).

         image               laser
           |                   |
         conv2d              conv1d
           |                   |
         conv2d                |
           |                   |
        flatten             flatten
           |                   |
         dense               dense
           |                   |
           --------merge--------
                     |
                   dense
                     |
                   dense
                     |
                   dense
                     |
                   state


        :param observations:
        :param is_training:
        :param l2_reg:
        :param batch_norm:
        :return state:
        """

        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=l2_reg)}
        initializers = {"w": tf.keras.initializers.he_normal()}

        # Camera branch
        camera_inputs = snt.SliceByDim(dims=[1], begin=[0], size=[self.obs_dim[0] * self.obs_dim[1] * 3])(observations)
        camera_inputs = tf.reshape(camera_inputs, [-1, self.obs_dim[0], self.obs_dim[1], 3])

        camera_conv1 = tf.layers.conv2d(camera_inputs, filters=32, kernel_size=3, strides=1, padding='valid',
                                        activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                        kernel_initializer=initializers["w"])

        if batch_norm:
            camera_conv1 = tf.layers.batch_normalization(camera_conv1, training=is_training)

        camera_conv2 = tf.layers.conv2d(camera_conv1, filters=64, kernel_size=3, strides=1, padding='valid',
                                        activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                        kernel_initializer=initializers["w"])
        if batch_norm:
            camera_conv2 = tf.layers.batch_normalization(camera_conv2, training=is_training)

        camera_flatten = snt.BatchFlatten()(camera_conv2)

        camera_dense1 = tf.layers.dense(camera_flatten, 64, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                        kernel_initializer=initializers["w"])

        # Laser branch
        laser_inputs = snt.SliceByDim(dims=[1], begin=[self.obs_dim[0] * self.obs_dim[1] * 3], size=[self.obs_dim[2]])(
            observations)
        laser_inputs = tf.reshape(laser_inputs, [-1, self.obs_dim[2], 1])

        laser_conv1 = tf.layers.conv1d(laser_inputs, filters=32, kernel_size=3, strides=1, padding='valid',
                                       activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])

        if batch_norm:
            laser_conv1 = tf.layers.batch_normalization(laser_conv1, training=is_training)


        laser_flatten = snt.BatchFlatten()(laser_conv1)

        laser_dense1 = tf.layers.dense(laser_flatten, 64, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])

        # Merged layers
        merge_input = tf.concat(values=[camera_dense1, laser_dense1], axis=1)
        merge_dense1 = tf.layers.dense(merge_input, 64, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])
        merge_dense2 = tf.layers.dense(merge_dense1, 32, activation=tf.nn.relu, kernel_regularizer=regularizers["w"],
                                       kernel_initializer=initializers["w"])

        state = tf.layers.dense(merge_dense2, self.state_dim, activation=None,
                                      kernel_regularizer=regularizers["w"], kernel_initializer=tf.initializers.zeros)

        merge_noise = lambda x: tf.cond(self.is_training, lambda: x + tf.random_normal(shape=tf.shape(x), stddev=1e-6),
                                        lambda: x)
        return merge_noise(state)

    def remember(self, observation, action, reward, done, sample_number):
        self.memory.store(observation, action, reward, done, sample_number) 

    def learn(self, use_action=False):
        with tqdm(total=self.memory.size) as pbar:
            t_loss = 0
            losses = np.zeros(6)
            nr_batches = self.memory.size // self.batch_size
            for _ in range(nr_batches):
                b1, b2, a1, a2, r1 = self.memory.sample_batch(self.batch_size, use_acts=use_action)

                # run the graph
                feed_dict = {self.obs_1:b1.get('obs'),
                             self.obs_2:b2.get('obs'),
                             self.obs_3:a1.get('obs'),
                             self.obs_4:a2.get('obs'),
                             self.obs_5:r1.get('obs'),
                             self.is_training:True}

                _, loss, l = self.sess.run([self.train_op, self.loss, self.losses], feed_dict=feed_dict)
                t_loss += loss
                losses += l
                pbar.update(self.batch_size)

        return t_loss / nr_batches, losses / nr_batches     
        
    def predict(self, observation):
        feed_dict = {self.obs_1:np.reshape(observation, (1, -1)), 
                     self.is_training:False}
        state = self.sess.run(self.s1, feed_dict=feed_dict)
        return state

    def predict_all(self, observations, batch_size=512):
        observations = observations.reshape(len(observations),-1)
        states = np.ndarray([len(observations),self.state_dim])
        num_batches = int(np.trunc((observations.shape[0]/batch_size)))

        for i in range(num_batches):
            states[ i * batch_size: (i+1) * (batch_size)] = self.sess.run(self.s1,
                                                                          feed_dict={
                                                                              self.obs_1: observations[ i * batch_size: (i+1) * (batch_size)],
                                                                              self.is_training: False
                                                                          })
        states[num_batches  * batch_size:] = self.sess.run(self.s1, feed_dict={
                                                                            self.obs_1: observations[num_batches * batch_size:],
                                                                            self.is_training: False
                                                                    })
        return states

    def get_memory_states(self):
        states = []
        mem = self.memory.get_all_samples().get('obs')

        states = self.predict_all(mem, batch_size=128)
        return states, self.memory.get_all_samples()

    def save_model(self):
        print("SRL Network storing model..........")
        return self.saver.save(self.sess, self.model_path)

    def load_model(self):
        print("SRL Network restoring data...........")
        self.saver.restore(self.sess, self.model_path)

    # Each prior is defined in a function 
    def temporal_coherence_prior(self, s_d):
        """
        E[||s_t+1 - s_t||_2]

        :param s_d: s_t+1 - s_t
        :return: expectation of the temporal coherence loss
        """
        return tf.reduce_mean(tf.norm(s_d, ord=2, axis=1)**2)

    def causality_prior(self, s1, s2):
        """
        E[e^(-||s1-s2||^2) | r_t2 != r_t1]

        :param s1: state s_t1
        :param s2: state s_t2
        :return: expectation of the causality loss
        """
        closs = tf.math.exp(-tf.norm(s1-s2, ord=2, axis=1)**2)

        return tf.reduce_mean(closs)

    def proportionality_prior(self, sd1, sd2):
        """
        E[(||s_t2+1 - s_t2||_2 - ||s_t1+1 - s_t1||_2)^2 | |r_t2+1 - r_t2| ~ |r_t2+1 - r_t2|]

        :param sd1: s_t1+1 - s_t1
        :param sd2: s_t2+1 - s_t2
        :return: expectation of the proportinality loss
        """
        ploss = (tf.norm(sd2, ord=2, axis=1) - tf.norm(sd1, ord=2, axis=1))**2

        return tf.reduce_mean(ploss)

    def repeatability_prior(self, s1, s2, s_d1, s_d2):
        """
        E[||(s_t2+1 - s_t2) - (s_t1+1 - s_t1)||_2)^2 * e^(-||s1-s2||^2) | |r_t2+1 - r_t2| ~ |r_t2+1 - r_t2|]


        :param s1: state s_t1
        :param s2: state s_t2
        :param s_d1: s_t1+1 - s_t1
        :param s_d2: s_t2+1 - s_t2
        :return: expectation of the repeatability loss
        """
        rloss1 = tf.math.exp(-tf.norm(s1-s2, ord=2, axis=1)**2)
        rloss2 = tf.norm(s_d2-s_d1, ord=2, axis=1)**2

        return tf.reduce_mean(rloss1*rloss2)


if __name__ == '__main__':
    from Logger import Logger
    import Plotter as plotter

    # Load saved observations from memory (samples collected by random exploring the large 4 walls environment --> see our paper)
    folder = 'training_data/observations_4walls_large.pkl'

    data = load_pickle(folder)
    print('Loaded {} data points'.format(len(data)))


    srl = StateRepresentation(obs_dim=[32, 24, 40], state_dim=5, batch_size=256, learning_rate=1e-3, seed=3,
                              continuelearning=False, usingros=False)


    # add all data to srl memory
    for d in data:
        srl.remember(reshape_observation(d[0]), d[1], d[2], d[3], d[4])

    epochs = 20
    loss_history = []
    use_action = False

    # training the encoder for 20 epochs
    for epoch in range(epochs):
        loss, losses = srl.learn(use_action=use_action)
        print('Finished epoch {}/{}. The loss this epoch was: {}'.format(epoch + 1, epochs, loss))
        print('temp_coh_loss: {} caus_loss: {} prop_loss: {} repeat_loss: {}'.format(losses[1], losses[2], losses[3],
                                                                                     losses[4]))

        loss_history.append(losses)

    # save the trained model
    srl.save_model()
    
    # compute all the states from the observations
    print('Finished training, now predicting all states')
    lg = Logger('./')
    gt = []
    for d in data:
        gt.append(d[0][40:45])
        lg.log("position_obs", d[0][40:45])
        lg.log("rewards", d[2])

    states, _ = srl.get_memory_states()

    # visualise the results
    print('Plotting the results')

    if use_action==False:
        name = "reward"
    else:
        name = "action"

    plotter.plot_all_srl(states=states, rews=srl.memory.get_all_samples().get('rews'), data_dict=lg.logDict,
                         loss_history=loss_history, trainingcycle=2, save=True, size=15000, name=name)
