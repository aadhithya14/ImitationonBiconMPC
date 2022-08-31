import os
import math
import numpy as np
import tensorflow as tf
import yaml
import json
import argparse
from baselines.common.tf_util import dense as lin
import baselines.common.tf_util as U
from utils.data_logging import Log
from scipy.stats import cauchy, t

class TrajRNN():

    def __init__(self, input_signal_dim, output_signal_dim, chunk_size, num_hidden=64, num_layers=3):
        self.input_signal_dim = input_signal_dim
        self.output_signal_dim = output_signal_dim
        self.chunk_size = chunk_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.make_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.reset()

    def make_graph(self, learning_rate=1e-2):
        #TODO: chuck vs full seq versions here -> unify
        self.input_sequence = tf.placeholder(tf.float32, [None, self.chunk_size, self.input_signal_dim])
        self.des_output_sequence = tf.placeholder(tf.float32, [None, self.chunk_size, self.output_signal_dim])

        cell = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=False) for i in range(self.num_layers)])
        self.initial_state = cell.zero_state(tf.shape(self.input_sequence)[0], dtype=tf.float32)
        self.seq_length = tf.fill([tf.shape(self.input_sequence)[0]], self.chunk_size)
        output_sequence, self.final_state = tf.nn.dynamic_rnn(
            cell, self.input_sequence, dtype=tf.float32, initial_state=self.initial_state, sequence_length=self.seq_length)

        output_sequence = tf.reshape(output_sequence, [-1, self.num_hidden])
        output_sequence = lin(output_sequence, 2 * self.output_signal_dim, 'output_layer')
        output_sequence = tf.reshape(output_sequence, [-1, self.chunk_size, 2 * self.output_signal_dim])


        def log_prob(mean, sigma, x):
            return U.sum(tf.log(1.0 / tf.sqrt(2.0 * np.pi * tf.square(sigma)) * tf.exp(-tf.square(x - mean) / (2.0 * tf.square(sigma)))), axis=-1)


        self.mean = tf.reshape(output_sequence[:, :, :self.output_signal_dim], [-1, self.chunk_size, self.output_signal_dim])
        sigma_output = tf.reshape(output_sequence[:, :, self.output_signal_dim:], [-1, self.chunk_size, self.output_signal_dim])
        self.sigma = tf.sqrt(tf.log(1.0 + tf.exp(sigma_output)))

        self.loss = tf.reduce_mean(-log_prob(self.mean, self.sigma, self.des_output_sequence))
        self.update = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


    def reset(self):
        self.internal_state = None
        self.steps_since_reset = 0

    def get_action(self, state, stohastic=True, combine=False, policy_mean=None, policy_sigma=None):
        input_sequence = np.zeros([1, self.chunk_size, self.input_signal_dim])
        input_sequence[0, 0] = state
        if self.internal_state is None or self.steps_since_reset == self.chunk_size:
            rnn_mean, rnn_sigma, self.internal_state = self.sess.run([self.mean, self.sigma, self.final_state], feed_dict={
                self.input_sequence: input_sequence, self.seq_length: np.array([1])})
            self.steps_since_reset = 1
        else:
            rnn_mean, rnn_sigma, self.internal_state = self.sess.run([self.mean, self.sigma, self.final_state], feed_dict={
                self.input_sequence: input_sequence, self.initial_state: self.internal_state, self.seq_length: np.array([1])})
            self.steps_since_reset += 1

        noise_mean = rnn_mean[0, 0]
        noise_sigma = rnn_sigma[0, 0]

        #print('noise', noise_mean, noise_sigma)
        #print('policy', policy_mean, policy_sigma)

        if combine:
            noise_mean = (noise_sigma ** 2 * policy_mean + policy_sigma ** 2 * noise_mean) / (noise_sigma ** 2 + policy_sigma ** 2)
            noise_sigma = np.sqrt(1 / (1 / noise_sigma ** 2 + 1 / policy_sigma ** 2))

        #print('result', noise_mean, noise_sigma)
        #print('after policy', policy_mean, policy_sigma)

        if stohastic:
            return np.random.normal(noise_mean, noise_sigma)
        else:
            return noise_mean

    def load_from_file(self, filename):
        # TODO: RNN should be in a scope other than root ('rnn')
        #traj_rnn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnn')
        #loader = tf.train.Saver(traj_rnn_variables)
        loader = tf.train.Saver()
        loader.restore(self.sess, filename)

    def save_to_file(self, filename):
        #traj_rnn_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnn')
        #saver = tf.train.Saver(traj_rnn_variables)
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    @staticmethod
    def make_dataset(root_folder, task_dirs, episodes_per_task, input_signal_dim, chunk_size, balance_task_data=True):
        # TODO: It should work for full sequences as well
        task_input_data = []
        task_output_data = []

        chunks_per_episode = int(400 / len(task_dirs))

        for task_dir in task_dirs:
            loaded_episodes = 0
            pack_num = 0
            task_folder = root_folder + task_dir
            if os.path.exists(task_folder + 'val_env_packs/'):
                collected_data_folder = task_folder + 'val_env_packs/'
            else:
                collected_data_folder = task_folder + 'env_packs/'
            task_input_data.append([])
            task_output_data.append([])

            packs = []
            for f in os.listdir(collected_data_folder):
                if f.endswith('json'):
                    packs.append(f)

            curr_pack_index = 0
            while loaded_episodes < episodes_per_task and curr_pack_index < len(packs):
                try:
                    with open(collected_data_folder + packs[curr_pack_index]) as f:
                        data = json.load(f)
                    states = data['states']
                    actions = data['actions']

                    for i in range(len(states)):
                        chunk_starts = np.random.randint(len(states[i]) - 1 - chunk_size + 1, size=chunks_per_episode)
                        for j in chunk_starts:
                            task_input_data[-1].append(states[i][j : j + chunk_size])
                            task_output_data[-1].append(actions[i][j : j + chunk_size])
                    pack_num += 1
                    loaded_episodes += len(states)
                except:
                    pass
                curr_pack_index += 1


        if balance_task_data:
            min_chunks = min([len(task_input_data[i]) for i in range(len(task_input_data))])
            for i in range(len(task_input_data)):
                task_input_data[i] = task_input_data[i][:min_chunks]
                task_output_data[i] = task_output_data[i][:min_chunks]

        input_dataset = []
        output_dataset = []
        for i in range(len(task_input_data)):
            input_dataset += task_input_data[i]
            output_dataset += task_output_data[i]

        input_dataset = np.array(input_dataset)
        output_dataset = np.array(output_dataset)

        if input_signal_dim != input_dataset.shape[2]:
            input_dataset = input_dataset[:, :, :input_signal_dim]

        return input_dataset, output_dataset

    def train(self, root_folder, task_dirs, episodes_per_task, batch_size, training_steps, training_folder, balance_task_data=True, log_update_freq=200):
        input_dataset, output_dataset = TrajRNN.make_dataset(root_folder, task_dirs, episodes_per_task, self.input_signal_dim, self.chunk_size, balance_task_data)
        log = Log(training_folder + 'training_log.json')

        for i in range(training_steps):
            batch_indices = np.random.randint(input_dataset.shape[0], size=batch_size)
            input_batch = input_dataset[batch_indices]
            output_batch = output_dataset[batch_indices]
            actual_loss = self.sess.run(self.loss, feed_dict={self.input_sequence: input_batch, self.des_output_sequence: output_batch})
            if math.isnan(actual_loss.item()) or actual_loss.item() > 50.0:
                continue
            _, actual_loss, actual_sigma = self.sess.run([self.update, self.loss, self.sigma], feed_dict={self.input_sequence: input_batch, self.des_output_sequence: output_batch})
            log.add('loss', actual_loss.item())
            log.add('sigma', np.linalg.norm(actual_sigma).item())
            if i % log_update_freq == 0:
                log.save()
                self.save_to_file(training_folder + 'latest')


def is_subset(subset, superset):
    for key, value in subset.items():
        if not key in superset:
            return False

        if isinstance(value, dict):
            if not is_subset(value, superset[key]):
                return False
        else:
            if value != superset[key]:
                return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rnn')
    args = parser.parse_args()
    training_spec = yaml.load(open(args.rnn + 'conf.yaml'))

    traj_rnn = TrajRNN(training_spec['input_signal_dim'], training_spec['output_signal_dim'], training_spec['chunk_size'], training_spec['num_hidden'], training_spec['num_layers'])

    if not 'task_dirs' in training_spec:
        assert 'num_tasks' in training_spec, 'Have to give EITHER task directories OR number of tasks for training.'
        num_tasks = training_spec['num_tasks']
        policies_folder = training_spec['root_folder'] + training_spec['policies_folder']
        contents = os.listdir(policies_folder)
        contents.sort()
        task_dirs = []
        for item in contents:
            if os.path.isdir(policies_folder + item):

                policy_folder = training_spec['policies_folder'] + item + '/'

                if 'env_params' in training_spec:
                    with open(training_spec['root_folder'] + policy_folder + 'conf.yaml') as f:
                        policy_conf = yaml.load(f)
                    policy_env_params = policy_conf['env_params'][0]['env_specific_params']

                    if not is_subset(training_spec['env_params'], policy_env_params):
                        continue

                task_dirs.append(policy_folder)
                if len(task_dirs) == num_tasks:
                    break
        training_spec['task_dirs'] = task_dirs
        print(task_dirs)

    traj_rnn.train(training_spec['root_folder'], training_spec['task_dirs'], training_spec['episodes_per_task'], training_spec['batch_size'], training_spec['training_steps'], args.rnn, training_spec['balance_task_data'])
    traj_rnn.save_to_file(args.rnn + 'final')
