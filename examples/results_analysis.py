import pickle
import os

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from dataclasses import dataclass
from acados_template import latexify_plot

latexify_plot()
@dataclass
class Results:
    reward: list
    t_wall: list
    cns_viol: list
    success: list
    max_cns_viol: list
    cost_cns_viol: list
    name: str
    state_list: list
    control_list: list


DATA_DIR = 'examples/mpc/temp-data'
FIGURE_DIR = 'figures/results'
DATE = '03_04'


def get_main_stats(name, METHOD, RTI, CONDESING, RL=False, snd_phase=False, **kwargs):
    mydict = {}
    for k, v in kwargs.items():
        mydict.update({k: v})

    if RL:
        FILENAME = f'{DATE}_trajscal_{mydict["traj_scaling"]}_rl_ppo_timed_l4casadi_data_quadrotor_traj_tracking.pkl'
    else:
        FILENAME = f'{DATE}_trajscal_{mydict["traj_scaling"]}_{METHOD}_N{mydict["N"]}_RTI_{RTI}_init_{mydict["initialization"]}_data_quadrotor_traj_tracking.pkl'
    if snd_phase:
        FILENAME = f'{DATE}_trajscal_{mydict["traj_scaling"]}_{METHOD}_M{mydict["M"]}N{mydict["N"]}_RTI_{RTI}_sndphase_{snd_phase}_init_{mydict["initialization"]}_initpept_{mydict["initialization_pept"]}_barrier_{mydict["barrier_parameter"]}_condensed_{CONDESING}_data_quadrotor_traj_tracking.pkl'

    path = os.path.join(DATA_DIR, FILENAME)

    with open(path, "rb") as f:
        results = pickle.load(f)

    reward_list = results['trajs_data']['reward']
    LEN_EPISODE = max([len(r) for r in reward_list])

    t_wall_list = results['trajs_data']['controller_data'][0]['t_wall']

    state_list = results['trajs_data']['state']
    control_list = results['trajs_data']['action']

    constraint_violation_list = []
    max_violation_list = []
    success_list = []
    constraint_penalization_list = []
    weight = 100  # 20 times higher than max weight in cost function, 0.02 is delta_T
    for idx, sim in enumerate(results['trajs_data']['info']):
        success = True
        if "solver_failed" in sim[-1].keys():
            if sim[-1]["solver_failed"]:
                success = False
        elif sim[-1]["current_step"] != LEN_EPISODE:
            success = False
        success_list.append(success)
        if success:
            viol = 0
            max_viol = -np.inf
            cost_constr_viol = []
            for sim_i in sim:
                try:
                    viol += sim_i['constraint_violation']
                    if sim_i['constraint_violation']:
                        max_viol = max(max_viol, max(sim_i['constraint_values'][sim_i['constraint_values'] >0]))
                except:
                    viol += any(sim_i['constraint_values'] >0) * 1
                if any(sim_i['constraint_values'] >0):
                    cost_constr_viol.append(weight * sum((sim_i['constraint_values'][sim_i['constraint_values'] > 1e-6])))

            constraint_violation_list.append(viol)
            max_violation_list.append(max_viol)
            constraint_penalization_list.append(np.array(cost_constr_viol))
        else:
            viol = 0
            max_viol = -np.inf
            cost_constr_viol = []
            for sim_i in sim[:-1]:
                try:
                    viol += sim_i['constraint_violation']
                    if sim_i['constraint_violation']:
                        max_viol = max(max_viol, max(sim_i['constraint_values'][sim_i['constraint_values'] >0]))
                except:
                    viol += any(sim_i['constraint_values'] >0) * 1
                if any(sim_i['constraint_values'] >0):
                    cost_constr_viol.append(weight * sum((sim_i['constraint_values'][sim_i['constraint_values'] > 1e-6])))

            constraint_violation_list.append(viol)
            max_violation_list.append(max_viol)
            constraint_penalization_list.append(np.array(cost_constr_viol))

    res = Results(reward=reward_list, t_wall=t_wall_list,
                  success=success_list, cns_viol=constraint_violation_list,
                  cost_cns_viol=constraint_penalization_list,
                  max_cns_viol=max_violation_list, name=name,
                  state_list=state_list, control_list=control_list)

    return res

if __name__ == "__main__":

    TRAJ_SCALING = 0.8  # 0.8, 1
    M = 10
    N = 20

    results = []
    kwargs = {}
    kwargs.update({"traj_scaling": TRAJ_SCALING, "N": N, "M": M})
    kwargs.update({"initialization": "shifting", "initialization_pept": "rollout"})

    kwargs.update({"N": 40})
    results.append(get_main_stats("RTI-40", "acados", True, False, **kwargs))
    kwargs.update({"N": 20})
    results.append(get_main_stats("RTI-20", "acados", True, False, **kwargs))
    kwargs.update({"barrier_parameter": "1e-4"})
    results.append(get_main_stats("PT-10-20-4", "acadosmp", True, True, snd_phase="pt", **kwargs))
    kwargs.update({"barrier_parameter": "1e-2"})
    results.append(get_main_stats("CLC-10-20-2", "acadosmp",  True, True, snd_phase="clc", **kwargs))
    results.append(get_main_stats("PEPT-10-20-2-r", "acadosmp", True, True, snd_phase="pept", **kwargs))
    results.append(get_main_stats("PPO-RL", "rl", False, False, RL=True, **kwargs))
    state_ref = np.load(f"{DATA_DIR}" + f"/state_ref_scaling_{TRAJ_SCALING}.npy")


    LEN_EPISODE = len(results[-1].reward[0])
    N_SIM = len(results[-1].reward)

    get_set_idx_success = lambda x: set(x.success * np.arange(1, N_SIM+1) - 1) - set([-1])
    is_success_fn = lambda success_idx, total_episodes, : [True if i in success_idx else False for i in range(total_episodes)]

    idx_all_success = set.intersection(*[get_set_idx_success(l) for l in results])
    is_success = is_success_fn(idx_all_success, N_SIM)

    t_sel_tot = lambda x : [t[:, 0] for t, s in zip(x.t_wall, is_success) if s == True]
    t_sel_prep = lambda x : [t[:, 1] for t, s in zip(x.t_wall, is_success) if s == True]
    t_sel_feed = lambda x : [t[:, 2] for t, s in zip(x.t_wall, is_success) if s == True]
    rew_sel_sum = lambda x : [sum(r) for r, s in zip(x.reward, is_success) if s == True]
    cost_cns_sel_sum = lambda x : [sum(c) for c, s in zip(x.cost_cns_viol, is_success) if s == True]
    cns_viol_ratio = lambda x : [c/LEN_EPISODE for c, s in zip(x.cns_viol, is_success) if s == True]
    max_cns_viol = lambda x : [c for c, s in zip(x.max_cns_viol, is_success) if s == True]

    labels = [l.name for l in results]
    plot_labels = labels


    # Print average runtime
    print("Average total runtime in milliseconds:")
    [print(l.name, np.round(np.array(t_sel_tot(l)).mean()*1e3, 2)) for l in (results)]
    print("Average feedback runtime in milliseconds:")
    [print(l.name, np.round(np.array(t_sel_feed(l)).mean()*1e3, 2)) for l in (results)]


    # Plot histogram with success
    lists = [l.success for l in results]
    counts = [ (lst.count(0), lst.count(1)) for lst in lists ]
    # Extract separate lists for plotting
    zeros_counts = [c[0] for c in counts]
    ones_counts = [c[1] for c in counts]
    x = np.arange(len(lists))  # the label locations
    width = 0.35  # width of the bars
    fig, ax = plt.subplots(figsize=(len(results)+1, 3))
    rects1 = ax.bar(x, 100/N_SIM*np.array(zeros_counts), width, label='Fail')
    ax.set_ylabel('Failures [\%]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()
    ax.grid(linestyle=':', alpha=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "success hist.pdf"))


    # Plot boxplot with tracking cost and violation cost
    total_rew = lambda x : [-np.nanmean(r) for r, s in zip(x.reward, is_success) if s == True]
    total_const = lambda x : [np.nanmean(r) for r, s in zip(x.cost_cns_viol, is_success) if s == True]

    fig = plt.figure(figsize=(3.5, 2.5))
    gs0 = gridspec.GridSpec(1, 2, figure=fig, left=0.25, bottom=0.15, wspace=0.11)
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=0.1, width_ratios=[1.8, 1])
    axs = []
    axs.append(fig.add_subplot(gs00[0, 0]))
    axs.append(fig.add_subplot(gs01[0, 0]))
    axs.append(fig.add_subplot(gs01[0, 1]))

    axs[0].boxplot([np.array(total_rew(l)) for l in reversed(results)], vert=False, showmeans=False)
    fig.text(0.39, 0.03, 'Reference tracking cost', ha='center', va='center', fontsize=7)
    axs[0].set_xlim(0, 2.8)

    axs[1].boxplot([np.array(total_const(l))[~np.isnan(total_const(l))] for l in reversed(results)], vert=False, showmeans=False, showfliers=False)
    axs[1].set_xlim(0, 0.04)
    axs[2].boxplot([np.array(total_const(l))[~np.isnan(total_const(l))] for l in reversed(results)], vert=False, showmeans=False, showfliers=False)
    axs[2].set_xlim(5, )
    plt.setp(axs[1].get_yticklabels(), visible=False)
    plt.setp(axs[2].get_yticklabels(), visible=False)
    axs[2].tick_params(axis='y', colors='w')

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axs[1].plot([1, 1], [0, 1], transform=axs[1].transAxes, **kwargs)
    axs[2].plot([0, 0], [0, 1], transform=axs[2].transAxes, **kwargs)

    fig.text(0.74, 0.03, 'Constraint violation cost', ha='center', va='center', fontsize=7)
    for ax in axs:
        ax.grid(linestyle=':', alpha=0.5)
    x = np.arange(1, len(lists)+1)  # the label locations
    axs[0].set_yticks(range(1,len(plot_labels)+1), reversed(plot_labels))
    axs[1].set_xticks(np.array([0, 0.015, 0.03]), ["0", "0.15", "0.3"])
    fig.savefig(os.path.join(FIGURE_DIR, "boxplot_tracking_violation_cost.pdf"), bbox_inches="tight", pad_inches=0)


    # Plot state trajectories (many trajectories shadowed)
    total_reward = lambda x : [-np.nanmean(r) if s == True else 9999 for r, s in zip(x.reward, is_success)]
    total_const = lambda x : [np.nanmean(r) if s == True else 9999 for r, s in zip(x.cost_cns_viol, is_success)]

    idx_max_viol_clc = np.argmin(total_reward(results[0]))
    [print(f"reward {r.name} : {total_reward(r)[idx_max_viol_clc]}") for r in results]
    state_list = [r.state_list[idx_max_viol_clc] for r in results]
    control_list = [r.control_list[idx_max_viol_clc] for r in results]

    nx = 12
    nu = 4
    stepsize = 0.02
    state_lb = [-2. , -1. , -2. , -1. ,  0. , -1. , -0.2, -0.2, -0.2, -1. , -1. ,  -1.]
    state_ub = [2. , 1. , 2. , 1. , 2. , 1. , 0.2, 0.2, 0.2, 1. , 1. , 1. ]
    control_lb = [0.029, 0.029, 0.029, 0.029]
    control_ub = [0.148, 0.148, 0.148, 0.148]
    control_ref = np.array([0.06615, 0.06615, 0.06615, 0.06615])
    times = np.linspace(0, stepsize * LEN_EPISODE, LEN_EPISODE)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Position-Velocity Projection XZ
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.7),)
    [axs[0].plot(np.array(s).T[0, :-1], np.array(s).T[4, :-1], color=colors[i], marker="", alpha=0.9, linewidth=1) for i, s in enumerate(state_list)]
    axs[0].plot(np.array(state_ref).T[0, :-1], np.array(state_ref).T[4, :-1], color='k', linestyle=':', alpha=0.3,)
    axs[0].set_ylabel('$p^\mathrm{z}$ (m)', labelpad=2)
    axs[0].set_xlabel('$p^\mathrm{x}$ (m)', labelpad=0.8)
    axs[0].set_xlim(-1.3, 1.3)

    [axs[1].plot(np.array(s).T[1, :-1], np.array(s).T[5, :-1], color=colors[i], marker='' , alpha=0.9, linewidth=1) for i, s in enumerate(state_list)]
    axs[1].plot(np.array(state_ref).T[1, :LEN_EPISODE//2], np.array(state_ref).T[5, :LEN_EPISODE//2], color='k', linestyle=':', alpha=0.3,)
    axs[1].axhline(state_lb[1], color="r", alpha=0.8, linestyle="--",)
    axs[1].axhline(state_ub[1], color="r", alpha=0.8, linestyle="--",)
    axs[1].axvline(state_lb[5], color="r", alpha=0.8, linestyle="--",)
    axs[1].axvline(state_ub[5], color="r", alpha=0.8, linestyle="--",)
    axs[1].set_ylabel('$v^\mathrm{z}$ (m/s)', labelpad=2)
    axs[1].set_xlabel('$v^\mathrm{x}$ (m/s)', labelpad=1)
    axs[1].set_xlim(-1.5, 1.5)
    fig.align_ylabels(axs)
    legend = fig.legend(plot_labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.01),
                        frameon=False, fontsize=7, handlelength=0.8)
    fig.subplots_adjust(top=0.88, hspace=0.34)
    plt.savefig(os.path.join(FIGURE_DIR, "pos_vel_XZ.pdf"), bbox_inches="tight", pad_inches=0)

    plt.show()
