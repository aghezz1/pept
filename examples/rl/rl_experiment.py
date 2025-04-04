'''This script tests the RL implementation.'''

from datetime import datetime
import os
import pickle
import shutil
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from safe_control_gym.envs.benchmark_env import Environment, Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=False, plot=True, n_episodes=1, n_steps=None, curr_path='.', save_data=False):
    '''Main function to run RL experiments.

    Args:
        gui (bool): Whether to display the gui.
        plot (bool): Whether to plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): How many steps to run the experiment.
        curr_path (str): The current relative path to the experiment folder.

    Returns:
        X_GOAL (np.ndarray): The goal (stabilization or reference trajectory) of the experiment.
        results (dict): The results of the experiment.
        metrics (dict): The metrics of the experiment.
    '''

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()

    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'
    if config.task == Environment.QUADROTOR:
        system = f'quadrotor_{str(config.task_config.quad_type)}D'
    else:
        system = config.task

    env_func = partial(make,
                       config.task,
                       **config.task_config)
    env = env_func(gui=gui)

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config,
                output_dir=curr_path + '/temp')

    # Load state_dict from trained.
    ctrl.load(f'{curr_path}/models/{config.algo}/{config.algo}_model_{system}_{task}.pt')

    # Remove temporary files and directories
    # shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)

    # Run experiment
    experiment = BaseExperiment(env, ctrl)
    trajs_data, metrics = experiment.run_evaluation(training=False, n_episodes=n_episodes, n_steps=n_steps)

    if plot:
        for i in range(len(trajs_data['obs'])):
            post_analysis(trajs_data['obs'][i], trajs_data['action'][i], env)

    ctrl.close()
    env.close()

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_rl_"

        with open(f'./temp-data/' + filename + f'{config.algo}_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
            pickle.dump(results, file)

    print('FINAL METRICS - ' + ', '.join([f'{key}: {value}' for key, value in metrics.items()]))


def post_analysis(state_stack, input_stack, env):
    '''Plots the input and states to determine MPC's success.

    Args:
        state_stack (ndarray): The list of observations of MPC in the latest run.
        input_stack (ndarray): The list of inputs of MPC in the latest run.
    '''
    model = env.symbolic
    stepsize = model.dt

    plot_length = np.min([np.shape(input_stack)[0], np.shape(state_stack)[0]])
    times = np.linspace(0, stepsize * plot_length, plot_length)

    reference = env.X_GOAL
    if env.TASK == Task.STABILIZATION:
        reference = np.tile(reference.reshape(1, model.nx), (plot_length, 1))
    # Plot states
    fig, axs = plt.subplots(model.nx, figsize=(6, 9), sharex=True)
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length], label='actual')
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='desired')
        axs[k].axhline(env.constraints.constraints[0].lower_bounds[k], color="r", alpha=0.5, linestyle="--")
        axs[k].axhline(env.constraints.constraints[0].upper_bounds[k], color="r", alpha=0.5, linestyle="--")
        axs[k].set(ylabel=env.STATE_LABELS[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[k].set_xlim(times[0], times[-1])
    axs[0].set_title('State Trajectories')
    axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()
    # Projection XZ, YZ
    fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    axs[0].plot(np.array(state_stack).transpose()[4, 0:plot_length],
                np.array(state_stack).transpose()[0, 0:plot_length], label='actual', marker='.')
    axs[0].plot(np.array(reference).transpose()[4, 0:plot_length],
                np.array(reference).transpose()[0, 0:plot_length], color='r', label='actual')
    axs[0].set_ylabel('X position [m]')
    axs[1].plot(np.array(state_stack).transpose()[4, 0:plot_length],
                np.array(state_stack).transpose()[2, 0:plot_length], label='actual', marker='.')
    axs[1].plot(np.array(reference).transpose()[4, 0:plot_length],
                np.array(reference).transpose()[2, 0:plot_length], color='r', label='actual')
    axs[1].set_ylabel('Y position [m]')
    axs[1].set_xlabel('Z position [m]')
    axs[-1].legend(ncol=1, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='center right')
    fig.tight_layout()
    # Plot inputs
    fig, axs = plt.subplots(model.nu, sharex=True)
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        control_k = np.array(input_stack).transpose()[k, 0:plot_length]
        if env.NORMALIZED_RL_ACTION_SPACE:
            control_k = env.denormalize_action(control_k)
            env.ACTION_UNITS[k] = "N"
        axs[k].plot(times, control_k)
        axs[k].set(ylabel=f'input {k}')
        axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[k].set_ylim(0.8*env.physical_action_bounds[0][k], 1.2*env.physical_action_bounds[1][k])
        axs[k].axhline(env.physical_action_bounds[0][k], color="r", alpha=0.5, linestyle="--")
        axs[k].axhline(env.physical_action_bounds[1][k], color="r", alpha=0.5, linestyle="--")
        axs[k].set_xlim(times[0], times[-1])
    axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    run()
