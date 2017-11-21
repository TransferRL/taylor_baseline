# Not formal tests
import sys
import os
import pickle as cPickcle
import lib.env.mountain_car
import lib.qlearning as ql
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from lib.instance_sampler import InstanceSampler

def generate_2D_sampler(num):
    s = InstanceSampler()
    result = []
    for i in range(num):
        result.append(s.getRandom2DInstance())
    return result


def generate_3D_sampler(num):
    s = InstanceSampler()
    result = []
    for i in range(num):
        result.append(s.getRandom3DInstance())
    return result

def generate_2D_optimal_samplers(num):
    result = []
    env = lib.env.mountain_car.MountainCarEnv()
    qlearning = ql.QLearning(env)
    qlearning.learn()
    for i in range(num):
        replay_mem = qlearning.play()
        result.append(replay_mem)
    return result

if __name__ == "__main__":
    with open('./data/optimal_instances.pkl', 'wb+') as f:
        cPickcle.dump(generate_2D_optimal_samplers(100), f);
    # with open('./data/2d_instances.pkl', "wb+") as f:
    #     cPickcle.dump(generate_2D_sampler(5000), f)
    # with open('./data/3d_instances.pkl', "wb+") as f:
    #     cPickcle.dump(generate_3D_sampler(5000), f)

    # Read example
    # Data format:
    # Arrays of instances. Each instance:
    # [state, action, next_state, reward, done]
    # with open('./data/2d_instances.pkl', "rb") as f:
    #     instances = cPickcle.load(f)
    #     count_terminations = [1 for ele in instances if ele[4]]
    #     print(len(count_terminations))
    # with open('./data/3d_instances.pkl', "rb") as f:
    #     instances = cPickcle.load(f)
    #     count_terminations = [1 for ele in instances if ele[4]]
    #     print(len(count_terminations))


    # Read example optimals samples
    # Data format:
    # Arrays of episodes data. Each episode consists all instances.
    with open('./data/optimal_instances.pkl', "rb") as f:
        episodes = cPickcle.load(f)
        lengths = [len(episode) for episode in episodes]
        print(np.asarray(lengths).sum())