from __future__ import print_function
import gym
import itertools
import matplotlib
import numpy as np
import tensorflow as tf
from lib.env.threedmountain_car import ThreeDMountainCarEnv
import lib.RandomAction
from lib.env.mountain_car import MountainCarEnv
import matplotlib.pyplot as plt
import os
import lib.qlearning as ql
import pickle
import deepq

# Create model
def neural_net(x, weights, biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def one_step_transition_model(learning_rate=0.1, n_hidden_1 = 32, n_hidden_2 = 32, num_input = 5, num_output = 4):

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_output]))
    }

    # Construct model
    logits = neural_net(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.losses.mean_squared_error(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    return loss_op, train_op, X, Y



def get_train_test_data(source_qlearn=True, source_env=MountainCarEnv(), target_env=ThreeDMountainCarEnv()):

    # source task
    if source_qlearn: # collect data from qlearning = true, collect data from random actions = false
        source_filename = './' + source_env.name + '_dsource_qlearn.npz'
        if os.path.isfile(source_filename):
            f_read = np.load(source_filename)
            dsource = f_read['dsource']

        else:
            model = deepq.models.mlp([64], layer_norm=True)
            act = deepq.learn(
                source_env,
                q_func=model,
                lr=1e-3,
                max_timesteps=40000,
                buffer_size=50000,
                exploration_fraction=0.1,
                exploration_final_eps=0.1,
                print_freq=1,
                param_noise=False
            )

            replay_memory = []  # reset
            for ep in range(100): # 100 episodes
                obs, done = source_env.reset(), False
                while not done:
                    n_obs, rew, done, _ = source_env.step(act(obs[None])[0])
                    replay_memory.append([obs, act(obs[None])[0], n_obs, rew, done])
                    obs = n_obs

            dsource = np.array(replay_memory)
            np.savez(source_filename, dsource=dsource)
            # with open('./data/q_learning.pkl', 'wb') as file:
            #     pickle.dump(qlearning_2d, file)
    else:
        source_filename = './' + source_env.name + '_dsource_random.npz'
        if os.path.isfile(source_filename):
            f_read = np.load(source_filename)
            dsource = f_read['dsource']
        else:
            qlearning_2d = lib.RandomAction.RandomAction(source_env)
            dsource = np.array(qlearning_2d.play())
            np.savez(source_filename, dsource=dsource)

    # target task
    target_filename = './' + target_env.name + '_dtarget_random.npz'
    if os.path.isfile(target_filename):
        f_read = np.load(target_filename)
        # print(f_read['dtarget'].shape)
        dtarget = f_read['dtarget']
    else:
        random_action_3d = lib.RandomAction.RandomAction(target_env)
        dtarget = np.array(random_action_3d.play())
        np.savez(target_filename, dtarget=dtarget)

    # Define the input function for training
    dsa = np.array([np.append(x[0], x[1]) for x in dtarget]) # dsa = d states actions
    dns = np.array([x[2] for x in dtarget]) # dns = d next states

    dsa_train = dsa[:-100]
    dns_train = dns[:-100]
    dsa_test = dsa[-100:]
    dns_test = dns[-100:]

    return dsa_train, dns_train, dsa_test, dns_test, dsource, dtarget


def train_model(num_steps=10000, batch_size=100, display_step=100, source_env=MountainCarEnv(), target_env=ThreeDMountainCarEnv()):
    loss_op, train_op, X, Y = one_step_transition_model()
    dsa_train, dns_train, dsa_test, dns_test, dsource, dtarget = get_train_test_data()
    batch_num = np.size(dsa_train, 0)

    init = tf.global_variables_initializer()
    loss = []

    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(num_steps):
            batch_x = dsa_train[(step * batch_size) % batch_num: (step * batch_size + batch_size) % batch_num, :]
            batch_y = dns_train[(step * batch_size) % batch_num: (step * batch_size + batch_size) % batch_num, :]

            # Run optimization op (backprop)
            loss_train, _ = sess.run([loss_op, train_op], feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0:
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss_train))
                loss.append(loss_train)

        print("Optimization Finished!")

        # test set
        loss_test = sess.run(loss_op, feed_dict={X: dsa_test, Y: dns_test})
        print("test loss is {}".format(loss_test))

        save_path = saver.save(sess, "./data/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

        # Find the mapping between source and target
        mc2d_env = MountainCarEnv()
        mc3d_env = ThreeDMountainCarEnv()
        mc2d_states = mc2d_env.observation_space.shape[0]  # 2
        mc3d_states = mc3d_env.observation_space.shape[0]  # 4
        mc2d_actions = mc2d_env.action_space.n  # 3
        mc3d_actions = mc3d_env.action_space.n  # 5

        mse_state_mappings = np.zeros((2,) * mc3d_states)  # 2 by 2 by 2 by 2
        mse_action_mappings = np.ndarray(shape=(mc3d_actions, mc2d_actions, mc3d_states * mc3d_states))  # 5 by 3 by 16
        mse_action_mappings.fill(-1)

        state_count = 0
        for s0 in range(mc2d_states):  # s0 is the first state of target states, x
            for s1 in range(mc2d_states):  # s1 is second state of target states, y
                for s2 in range(mc2d_states):  # s2 is third state of target states, x_dot
                    for s3 in range(mc2d_states):  # s3 is fourth state of target states, y_dot

                        state_losses = []

                        for a_mc3d in range(mc3d_actions):
                            for a_mc2d in range(mc2d_actions):
                                states = np.array([x[0] for x in dsource if x[1] == a_mc2d])
                                actions = np.array([x[1] for x in dsource if x[1] == a_mc2d])
                                n_states = np.array([x[2] for x in dsource if x[1] == a_mc2d])

                                if (states.size == 0) or (actions.size == 0) or (n_states.size == 0):
                                    print('this happened..')  # TODO
                                    # mse_action_mappings[a_mc3d, a_mc2d, state_count] = 0
                                    continue

                                # transform to dsource_trans
                                actions_trans = np.ndarray(shape=(actions.size,))
                                actions_trans.fill(a_mc3d)
                                input_trans = np.array(
                                    [states[:, s0], states[:, s1], states[:, s2], states[:, s3], actions_trans]).T
                                # input_trans = [states_trans, actions]
                                n_states_trans = np.array(
                                    [n_states[:, s0], n_states[:, s1], n_states[:, s2], n_states[:, s3]]).T

                                # calculate mapping error
                                loss_mapping = sess.run(loss_op, feed_dict={X: input_trans, Y: n_states_trans})
                                # print('loss_mapping is {}'.format(loss_mapping))

                                state_losses.append(loss_mapping)
                                mse_action_mappings[a_mc3d, a_mc2d, state_count] = loss_mapping

                        mse_state_mappings[s0, s1, s2, s3] = np.mean(state_losses)
                        state_count += 1

        # mse_action_mappings_result = [[np.mean(mse_action_mappings[a_mc3d, a_mc2d, :]) for a_mc2d in range(mc2d_actions)] for a_mc3d in range(mc3d_actions)]

        mse_action_mappings_result = np.zeros((mc3d_actions, mc2d_actions))
        for a_mc3d in range(mc3d_actions):
            for a_mc2d in range(mc2d_actions):
                losses_act = []
                for s in range(mc3d_states * mc3d_states):
                    if mse_action_mappings[a_mc3d, a_mc2d, s] != -1:
                        # print(mse_action_mappings[a_mc3d, a_mc2d, s])
                        losses_act.append(mse_action_mappings[a_mc3d, a_mc2d, s])
                mse_action_mappings_result[a_mc3d, a_mc2d] = np.mean(losses_act)

        print('action mapping: {}'.format(mse_action_mappings_result))
        print('state mapping {}'.format(mse_state_mappings))

        print('x,x,x,x: {}'.format(mse_state_mappings[0][0][0][0]))
        print('x,x,x,x_dot: {}'.format(mse_state_mappings[0][0][0][1]))
        print('x,x,x_dot,x: {}'.format(mse_state_mappings[0][0][1][0]))
        print('x,x,x_dot,x_dot: {}'.format(mse_state_mappings[0][0][1][1]))
        print('x,x_dot,x,x: {}'.format(mse_state_mappings[0][1][0][0]))
        print('x,x_dot,x,x_dot: {}'.format(mse_state_mappings[0][1][0][1]))
        print('x,x_dot,x_dot,x: {}'.format(mse_state_mappings[0][1][1][0]))
        print('x,x_dot,x_dot,x_dot: {}'.format(mse_state_mappings[0][1][1][1]))
        print('x_dot,x,x,x: {}'.format(mse_state_mappings[1][0][0][0]))
        print('x_dot,x,x,x_dot: {}'.format(mse_state_mappings[1][0][1][0]))
        print('x_dot,x,x_dot,x: {}'.format(mse_state_mappings[1][0][1][1]))
        print('x_dot,x,x_dot,x_dot: {}'.format(mse_state_mappings[1][1][0][0]))
        print('x_dot,x_dot,x,x: {}'.format(mse_state_mappings[1][0][0][1]))
        print('x_dot,x_dot,x,x_dot: {}'.format(mse_state_mappings[1][1][0][1]))
        print('x_dot,x_dot,x_dot,x: {}'.format(mse_state_mappings[1][1][1][0]))
        print('x_dot,x_dot,x_dot,x_dot: {}'.format(mse_state_mappings[1][1][1][1]))

        with open('./data/mse_state_mappings.pkl', 'wb') as file:
            pickle.dump(mse_state_mappings, file)

        with open('./data/mse_action_mappings.pkl', 'wb') as file:
            pickle.dump(mse_action_mappings, file)

        print("Done exporting MSE file")


if __name__ == '__main__':
    train_model(num_steps=10000, batch_size=100, display_step=100)

