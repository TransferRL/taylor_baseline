import gym

import deepq
from lib.env.threedmountain_car import ThreeDMountainCarEnv


def main():
    # env = gym.make("MountainCar-v0")
    env = ThreeDMountainCarEnv()
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
        param_noise=False
    )
    print("Saving model to mountaincar_model_working.pkl")
    act.save("mountaincar_model_working.pkl")


if __name__ == '__main__':
    main()
