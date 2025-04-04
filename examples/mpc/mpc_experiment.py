'''An MPC and Linear MPC example.'''

import os
import pickle
from functools import partial

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.experiments.base_experiment import BaseExperiment
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(gui=False, plot=True, n_episodes=1, n_steps=None, save_data=False):
    '''The main function running MPC and Linear MPC experiments.

    Args:
        gui (bool): Whether to display the gui.
        plot (bool): Whether to plot graphs.
        n_episodes (int): The number of episodes to execute.
        n_steps (int): The total number of steps to execute.
        save_data (bool): Whether to save the collected experiment data.
    '''

    # Create the configuration dictionary.
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config
                       )
    env = env_func(gui=gui)

    # Create controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config
                )

    # Run the experiment.
    experiment = BaseExperiment(env=env, ctrl=ctrl)
    trajs_data, metrics = experiment.run_evaluation(training=False, n_episodes=n_episodes, n_steps=n_steps)

    if plot:
        post_analysis(trajs_data['obs'][0], trajs_data['action'][0], env)

    ctrl.close()
    env.close()

    if config.algo == 'mpc_acados':
        filename = f"{datetime.now().strftime('%d_%m')}_trajscal_{config.task_config.task_info.trajectory_scale}_acados_N{config.algo_config.horizon}_RTI_{config.algo_config.use_RTI}_init_{config.algo_config.initialization}"
    elif config.algo == 'mpc_acados_m':
        if config.algo_config.second_phase.method == "rl":
            filename = f"{datetime.now().strftime('%d_%m')}_trajscal_{config.task_config.task_info.trajectory_scale}_rl_ppo_timed_l4casadi"
        else:
            filename = f"{datetime.now().strftime('%d_%m')}_trajscal_{config.task_config.task_info.trajectory_scale}_acadosmp_M{config.algo_config.short_horizon}N{config.algo_config.long_horizon}_RTI_{config.algo_config.use_RTI}_sndphase_{config.algo_config.second_phase.method}_init_{config.algo_config.second_phase.initialization}_initpept_{config.algo_config.second_phase.initialization_pept}_barrier_{config.algo_config.second_phase.barrier_parameter}_condensed_{config.algo_config.with_condensing_terminal_value}"
    else:
        filename = f"{config.algo}"

    if save_data:
        results = {'trajs_data': trajs_data, 'metrics': metrics}
        path_dir = os.path.dirname('./temp-data/')
        os.makedirs(path_dir, exist_ok=True)
        np.save(f'./temp-data/' + f'state_ref_scaling_{config.task_config.task_info.trajectory_scale}', env.X_GOAL)
        with open(f'./temp-data/' + filename + f'_data_{config.task}_{config.task_config.task}.pkl', 'wb') as file:
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
    fig, axs = plt.subplots(model.nx//2, 2, figsize=(7, 5), sharex=True)
    axs = axs.flatten()
    state_labels = ["$x$", "$\dot{x}$", "$y$", "$\dot{y}$", "$z$", "$\dot{z}$", "$\phi$", "$\\theta$", "$\psi$", "$p$", "$q$", "$r$"]
    for k in range(model.nx):
        axs[k].plot(times, np.array(state_stack).transpose()[k, 0:plot_length])
        axs[k].plot(times, reference.transpose()[k, 0:plot_length], color='r', label='ref', alpha=0.5)
        axs[k].axhline(env.constraints.constraints[0].lower_bounds[k], color="k", alpha=0.5, linestyle="--")
        axs[k].axhline(env.constraints.constraints[0].upper_bounds[k], color="k", alpha=0.5, linestyle="--")
        axs[k].set(ylabel=state_labels[k] + f'\n[{env.STATE_UNITS[k]}]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[k].set_xlim(times[0], times[-1])
    # axs[0].set_title('State Trajectories')
    # axs[-1].legend(ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='lower right')
    axs[-1].set(xlabel='time (sec)')
    axs[-2].set(xlabel='time (sec)')
    fig.tight_layout()
    # Projection XZ, YZ
    fig, axs = plt.subplots(2, 1, figsize=(4, 4),)
    axs[0].plot(
                np.array(state_stack).transpose()[0, 0:plot_length],
        np.array(state_stack).transpose()[4, 0:plot_length],
                label='actual', marker='.')
    axs[0].plot(
                np.array(reference).transpose()[0, 0:plot_length],
        np.array(reference).transpose()[4, 0:plot_length],
                color='r', label='actual')
    axs[0].set_ylabel('Z position [m]')
    axs[0].set_xlabel('X position [m]')
    axs[1].plot(
                np.array(state_stack).transpose()[2, 0:plot_length],
        np.array(state_stack).transpose()[4, 0:plot_length],
                label='actual', marker='.')
    axs[1].plot(
                np.array(reference).transpose()[2, 0:plot_length],
        np.array(reference).transpose()[4, 0:plot_length],
                color='r', label='actual')
    axs[1].set_xlabel('Y position [m]')
    axs[1].set_ylabel('Z position [m]')
    fig.align_ylabels(axs)
    # axs[-1].legend(ncol=1, bbox_transform=fig.transFigure, bbox_to_anchor=(1, 0), loc='center right')
    fig.tight_layout()
    # Plot inputs
    fig, axs = plt.subplots(model.nu, sharex=True, figsize=(3.5, 3))
    if model.nu == 1:
        axs = [axs]
    for k in range(model.nu):
        axs[k].plot(times, np.array(input_stack).transpose()[k, 0:plot_length])
        # axs[k].set(ylabel=f'input {k}')
        # axs[k].set(ylabel=env.ACTION_LABELS[k] + f'\n[{env.ACTION_UNITS[k]}]')
        axs[k].set(ylabel=f'$\\tau_{k+1}$ [N]')
        axs[k].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[k].set_ylim(0.8*env.physical_action_bounds[0][k], 1.2*env.physical_action_bounds[1][k])
        axs[k].axhline(env.physical_action_bounds[0][k], color="r", alpha=0.5, linestyle="--")
        axs[k].axhline(env.physical_action_bounds[1][k], color="r", alpha=0.5, linestyle="--")
        axs[k].set_xlim(times[0], times[-1])
    # axs[0].set_title('Input Trajectories')
    axs[-1].set(xlabel='time (sec)')
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    from acados_template import latexify_plot
    latexify_plot()
    run()
