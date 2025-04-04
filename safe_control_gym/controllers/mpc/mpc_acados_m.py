'''Model Predictive Control using Acados.'''

from copy import deepcopy
from functools import partial

import os
import time
from typing import Optional
import casadi as cs
import numpy as np
import scipy
from termcolor import colored
import l4casadi as l4c
import torch


from safe_control_gym.controllers.mpc.mpc import MPC
from safe_control_gym.controllers.mpc.mpc_utils import set_acados_constraint_bound
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import timing
from munch import Munch

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosMultiphaseOcp, AcadosSim, AcadosSimSolver
    from acados_template.mpc_utils import detect_constraint_structure
except ImportError as e:
    print(colored(f'Error: {e}', 'red'))
    print(colored('acados not installed, cannot use acados-based controller. Exiting.', 'red'))
    print(colored('- To build and install acados, follow the instructions at https://docs.acados.org/installation/index.html', 'yellow'))
    print(colored('- To set up the acados python interface, follow the instructions at https://docs.acados.org/python_interface/index.html', 'yellow'))
    print()
    exit()


def load_policy_and_convert_to_l4casadi_model():
    curr_path = os.path.join(os.getcwd(), '../rl')
    algo_name = "ppo"
    # Hard copy of algorithm and task to recreate the env linked with PPO and 3D drone tracking
    algo_config = Munch({'hidden_dim': 128, 'activation': 'tanh', 'norm_obs': False, 'norm_reward': False, 'clip_obs': 10, 'clip_reward': 10, 'gamma': 0.99, 'use_gae': True, 'gae_lambda': 0.95, 'use_clipped_value': False, 'clip_param': 0.2, 'target_kl': 0.01, 'entropy_coef': 0.01, 'opt_epochs': 20, 'mini_batch_size': 256, 'actor_lr': 0.001, 'critic_lr': 0.001, 'max_grad_norm': 0.5, 'max_env_steps': 1000000, 'num_workers': 1, 'rollout_batch_size': 4, 'rollout_steps': 1000, 'deque_size': 10, 'eval_batch_size': 10, 'log_interval': 10000, 'save_interval': 0, 'num_checkpoints': 0, 'eval_interval': 10000, 'eval_save_best': True, 'tensorboard': False, 'training': False})
    task_config = Munch({'info_in_reset': True, 'ctrl_freq': 50, 'pyb_freq': 1000, 'physics': 'pyb', 'gui': False, 'quad_type': 3, 'normalized_rl_action_space': True, 'episode_len_sec': 5, 'init_state': Munch({'init_x': 0.4, 'init_x_dot': 0, 'init_y': 0.4, 'init_y_dot': 0, 'init_z': 1.4, 'init_z_dot': 0, 'init_phi': 0, 'init_theta': 0, 'init_psi': 0, 'init_p': 0, 'init_q': 0, 'init_r': 0}), 'randomized_init': False, 'init_state_randomization_info': Munch({'init_x': Munch({'distrib': 'uniform', 'low': -2, 'high': 2}), 'init_x_dot': Munch({'distrib': 'uniform', 'low': -1, 'high': 1}), 'init_y': Munch({'distrib': 'uniform', 'low': -2, 'high': 2}), 'init_y_dot': Munch({'distrib': 'uniform', 'low': -1, 'high': 1}), 'init_z': Munch({'distrib': 'uniform', 'low': 0.3, 'high': 2}), 'init_z_dot': Munch({'distrib': 'uniform', 'low': -1, 'high': 1}), 'init_phi': Munch({'distrib': 'uniform', 'low': -0.2, 'high': 0.2}), 'init_theta': Munch({'distrib': 'uniform', 'low': -0.2, 'high': 0.2}), 'init_psi': Munch({'distrib': 'uniform', 'low': -0.2, 'high': 0.2}), 'init_p': Munch({'distrib': 'uniform', 'low': -1, 'high': 1}), 'init_q': Munch({'distrib': 'uniform', 'low': -1, 'high': 1}), 'init_r': Munch({'distrib': 'uniform', 'low': -1, 'high': 1})}), 'inertial_prop': Munch({'M': 0.027, 'Ixx': 1.4e-05, 'Iyy': 1.4e-05, 'Izz': 2.17e-05}), 'randomized_inertial_prop': False, 'inertial_prop_randomization_info': None, 'task': 'traj_tracking', 'task_info': Munch({'trajectory_type': 'figure8', 'num_cycles': 1, 'trajectory_plane': 'xz', 'trajectory_position_offset': [0, 1], 'trajectory_scale': 1, 'proj_point': [0, 0, 0.5], 'proj_normal': [0, 1, 1]}), 'cost': 'rl_reward', 'disturbances': None, 'adversary_disturbance': None, 'adversary_disturbance_offset': 0.0, 'adversary_disturbance_scale': 0.01, 'constraints': [Munch({'constraint_form': 'default_constraint', 'constrained_variable': 'state', 'upper_bounds': [2, 1, 2, 1, 2, 1, 0.2, 0.2, 0.2, 1, 1, 1], 'lower_bounds': [-2, -1, -2, -1, 0, -1, -0.2, -0.2, -0.2, -1, -1, -1]}), Munch({'constraint_form': 'default_constraint', 'constrained_variable': 'input', 'upper_bounds': [0.148, 0.148, 0.148, 0.148], 'lower_bounds': [0.029, 0.029, 0.029, 0.029]})], 'done_on_violation': False, 'use_constraint_penalty': False, 'constraint_penalty': -1, 'verbose': False, 'norm_act_scale': 0.1, 'obs_goal_horizon': 1, 'rew_state_weight': [1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01], 'rew_act_weight': 0.0001, 'rew_exponential': True, 'done_on_out_of_bound': True, 'seed': 1337, 'info_mse_metric_state_weight': [1, 0.01, 1, 0.01, 1, 0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01]})

    fac = ConfigFactory()
    config = fac.merge()
    env_func = partial(make,
                      config.task,
                        **task_config)
    env = env_func(gui=False)

    task = 'track'
    system = 'quadrotor_3D'
    # Setup controller.
    ctrl = make(algo_name,
                env_func,
                **algo_config,
                output_dir=curr_path + '/temp')
    # Load state_dict from trained.
    ctrl.load(f'{curr_path}/models/{algo_name}/{algo_name}_model_{system}_{task}.pt')
    pytorch_model = ctrl.agent.ac.actor.pi_net
    l4c_model = l4c.L4CasADi(pytorch_model, device='cpu')
    x_sym = cs.MX.sym('x', 1, 24)
    y_sym = env.denormalize_action(l4c_model(x_sym))
    f = cs.Function('y', [x_sym], [y_sym])
    return f, l4c_model, pytorch_model

def rk_discrete_casadi_expr(f, X, U, dt):
    '''Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        X (int): state var.
        U (int): input var.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    '''

    # Runge-Kutta 4 integration
    k1 = f(X, U)
    k2 = f(X + dt / 2 * k1, U)
    k3 = f(X + dt / 2 * k2, U)
    k4 = f(X + dt * k3, U)
    x_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next

def rk_discrete_eval(f, X, U, dt):
    # Runge-Kutta 4 integration
    k1 = f(X, U)
    k2 = f(X + dt / 2 * k1, U)
    k3 = f(X + dt / 2 * k2, U)
    k4 = f(X + dt * k3, U)
    x_next = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next.full().squeeze()

def euler_discrete_casadi_expr(f, X, U, dt):
    '''Runge Kutta discretization for the function.

    Args:
        f (casadi function): Function to discretize.
        X (int): state var.
        U (int): input var.
        dt (float): discretization time.

    Return:
        x_next (casadi function?):
    '''

    # Runge-Kutta 4 integration
    k1 = f(X, U)
    x_next = X + dt * k1

    return x_next

def formulate_constraint_as_log_barrier(
        ocp: AcadosOcp,
        constr_expr: cs.MX,
        weight: float,
        upper_bound: Optional[float],
        lower_bound: Optional[float],
        residual_name: str = "new_residual",
        constraint_type: str = "path",
    ) -> None:
    """
    Formulate a constraint as an log-barrier term and add it to the current cost.
    """
    from scipy.linalg import block_diag
    import casadi as ca

    casadi_symbol = ocp.model.get_casadi_symbol()

    if upper_bound is None and lower_bound is None:
        raise ValueError("Either upper or lower bound must be provided.")

    # compute violation expression
    log_barrier_input = casadi_symbol("log_barrier_input", 0, 1)
    if upper_bound is not None:
        # log_barrier_input = ca.vertcat(log_barrier_input, constr_expr - upper_bound)
        log_barrier_input = ca.vertcat(log_barrier_input, -constr_expr + upper_bound)
    if lower_bound is not None:
        # log_barrier_input = ca.vertcat(log_barrier_input, lower_bound - constr_expr)
        log_barrier_input = ca.vertcat(log_barrier_input, -lower_bound + constr_expr)
    y_ref_new = np.zeros((log_barrier_input.shape[0],))

    # add penalty as cost
    if constraint_type == "path":
        ocp.cost.yref = np.concatenate((ocp.cost.yref, y_ref_new))
        ocp.model.cost_y_expr = ca.vertcat(ocp.model.cost_y_expr, log_barrier_input)
        if ocp.cost.cost_type == "CONVEX_OVER_NONLINEAR":
            new_residual = casadi_symbol(residual_name, log_barrier_input.shape)
            ocp.model.cost_r_in_psi_expr = ca.vertcat(ocp.model.cost_r_in_psi_expr, new_residual)
            for i in range(log_barrier_input.shape[0]):
                ocp.model.cost_psi_expr -= weight * ca.log(new_residual[i])
        # elif ocp.cost.cost_type == "EXTERNAL":
        #     ocp.model.cost_expr_ext_cost += .5 * weight * violation_expr**2
        else:
            raise NotImplementedError(f"formulate_constraint_as_L2_penalty not implemented for path cost with cost_type {ocp.cost.cost_type}.")
    elif constraint_type == "initial":
        raise NotImplementedError("TODO!")
    elif constraint_type == "terminal":
        ocp.cost.yref_e = np.concatenate((ocp.cost.yref_e, y_ref_new))
        ocp.model.cost_y_expr_e = ca.vertcat(ocp.model.cost_y_expr_e, log_barrier_input)
        if ocp.cost.cost_type_e == "CONVEX_OVER_NONLINEAR":
            new_residual = casadi_symbol(residual_name, log_barrier_input.shape)
            ocp.model.cost_r_in_psi_expr_e = ca.vertcat(ocp.model.cost_r_in_psi_expr_e, new_residual)
            for i in range(log_barrier_input.shape[0]):
                ocp.model.cost_psi_expr_e -= weight * ca.log(new_residual[i])
    return ocp

class MPCAcadosMultiPhase(MPC):
    '''MPC with full nonlinear model.'''

    def __init__(
            self,
            # runner args
            env_func,
            long_horizon: int = 10,
            short_horizon: int = 5,
            q_mpc: list = [1],
            r_mpc: list = [1],
            initial_guess_t0: Munch = Munch(),
            use_RTI: bool = False,
            with_condensing_terminal_value: bool = False,
            second_phase: Munch = Munch(),
            # shared/base args
            soft_constraints: bool = False,
            soft_penalty: float = 10000,
            terminate_run_on_done: bool = True,
            constraint_tol: float = 1e-6,
            output_dir: str = 'results/temp',
            additional_constraints: list = None,
            use_gpu: bool = False,
            seed: int = 0,
            **kwargs
    ):
        '''Creates task and controller.

        Args:
            env_func (Callable): function to instantiate task/environment.
            horizon (int): mpc planning horizon.
            q_mpc (list): diagonals of state cost weight.
            r_mpc (list): diagonals of input/action cost weight.
            warmstart (bool): if to initialize from previous iteration.
            soft_constraints (bool): Formulate the constraints as soft constraints.
            soft_penalty (float): Penalty added to acados formulation for soft constraints.
            terminate_run_on_done (bool): Terminate the run when the environment returns done or not.
            constraint_tol (float): Tolerance to add the the constraint as sometimes solvers are not exact.
            output_dir (str): output directory to write logs and results.
            additional_constraints (list): List of additional constraints
            use_gpu (bool): False (use cpu) True (use cuda).
            seed (int): random seed.
            use_RTI (bool): Real-time iteration for acados.
        '''
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__.update({k: v})

        assert (initial_guess_t0.initialization_type in ["tracking_goal", "lqr", "ipopt", "policy"])
        assert (second_phase.method in ["pept", "clc", "pt", "rl"])
        assert (second_phase.initialization in ["warm_starting", "shifting"])
        assert (second_phase.initialization_pept in ["rollout", "stage_wise"])

        super().__init__(
            env_func,
            # horizon=horizon,
            q_mpc=q_mpc,
            r_mpc=r_mpc,
            warmstart=initial_guess_t0.initialize,
            soft_constraints=soft_constraints,
            soft_penalty=soft_penalty,
            terminate_run_on_done=terminate_run_on_done,
            constraint_tol=constraint_tol,
            output_dir=output_dir,
            additional_constraints=additional_constraints,
            compute_initial_guess_method=initial_guess_t0.initialization_type,
            use_gpu=use_gpu,
            seed=seed,
            **kwargs
        )
        # acados settings
        self.use_RTI = use_RTI
        self.ocp_solver_exist = False
        self.T = long_horizon
        self.short_horizon = short_horizon
        self.with_condesing_terminal_value = with_condensing_terminal_value
        self.pi_casadi_fn, self.model_l4casadi, self.pytorch_model = load_policy_and_convert_to_l4casadi_model()
        self.pi_eval = lambda x: self.pi_casadi_fn(x).full().squeeze()
        self.second_phase = second_phase

    @timing
    def reset(self):
        '''Prepares for training or evaluation.'''
        print(colored('Resetting MPC', 'green'))
        self.x_guess = None
        self.u_guess = None
        super().reset()

        if self.second_phase.method == "rl":
            # Warm start l4casadi
            self.setup_acados_model()
            [self.pi_casadi_fn(np.ones(self.acados_model.x.shape[0]*2)) for _ in range(10)]
        else:
            if not self.ocp_solver_exist:
                self.ocp_solver_exist = True
                self.acados_model = None
                self.ocp = None
                self.acados_ocp_solver = None
                # Dynamics model.
                self.setup_acados_model()
                # Acados optimizer.
                self.setup_acados_optimizer()
                self.acados_ocp_solver = AcadosOcpSolver(self.ocp)
                # Warm start l4casadi
                [self.acados_ocp_solver.solve() for _ in range(3)]
                [self.pi_casadi_fn(np.ones(self.acados_model.x.shape[0]*2)) for _ in range(10)]

    def setup_acados_model(self) -> AcadosModel:
        '''Sets up symbolic model for acados.

        Returns:
            acados_model (AcadosModel): acados model object.

        Other options to set up the model:
        f_expl = self.model.x_dot (explicit continuous-time dynamics)
        f_impl = self.model.x_dot_acados - f_expl (implicit continuous-time dynamics)
        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        '''

        acados_model = AcadosModel()
        acados_model.x = self.model.x_sym
        acados_model.u = self.model.u_sym
        acados_model.name = self.env.NAME

        acados_model.f_expl_expr = self.model.x_dot

        # store meta information # NOTE: unit is missing
        acados_model.x_labels = self.env.STATE_LABELS
        acados_model.u_labels = self.env.ACTION_LABELS
        acados_model.t_label = 'time'

        self.acados_model = acados_model

    @timing
    def compute_initial_guess(self, init_state, goal_states=None):
        '''Use IPOPT to get an initial guess of the solution.

        Args:
            init_state (ndarray): Initial state.
            goal_states (ndarray): Goal states.
        '''
        x_val, u_val = super().compute_initial_guess(init_state, goal_states)
        self.x_guess = x_val
        self.u_guess = u_val

    def setup_acados_optimizer(self):
        '''Sets up nonlinear optimization problem.'''

        self.setup_multiphase_ocp()
        self.setup_acados_simulator()

    def setup_acados_simulator(self):
        sim = AcadosSim()
        sim.model.f_expl_expr = self.model.x_dot
        sim.model.x = self.model.x_sym
        sim.model.xdot = cs.MX.sym("xdot", sim.model.x.shape)
        sim.model.f_impl_expr = sim.model.xdot - sim.model.f_expl_expr
        sim.model.u = self.model.u_sym
        sim.model.name = "quadrotor_sim"
        sim.solver_options.T = self.dt
        sim.solver_options.integrator_type = 'ERK'
        self.acados_integrator = AcadosSimSolver(sim)

    def setup_multiphase_ocp(self):
        second_horizon = int(self.T-self.short_horizon)
        N_list = [self.short_horizon, second_horizon]

        ocp = AcadosMultiphaseOcp(N_list=N_list)
        phase_0 = self.setup_first_phase_ocp()
        phase_1 = self.setup_second_phase_ocp()

        # set phases
        ocp.set_phase(phase_0, phase_idx=0)
        ocp.set_phase(phase_1, phase_idx=1)
        if self.second_phase.method == "clc":
            ocp.mocp_opts.integrator_type = ['ERK', 'DISCRETE']
        else:
            ocp.mocp_opts.integrator_type = ['ERK', 'ERK']
        # set up solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        if self.with_condesing_terminal_value:
            ocp.solver_options.qp_solver_cond_block_size = [1] * self.short_horizon + [second_horizon]
            ocp.solver_options.qp_solver_cond_N = self.short_horizon
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP' if not self.use_RTI else 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.levenberg_marquardt = 1.
        if not self.use_RTI:
            ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH'  # 'MERIT_BACKTRACKING'
        ocp.solver_options.tf = self.T * self.dt  # prediction horizon
        if self.second_phase.method == "clc":
            ocp.solver_options.model_external_shared_lib_dir = self.model_l4casadi.shared_lib_dir
            ocp.solver_options.model_external_shared_lib_name = self.model_l4casadi.name
        ocp.code_export_directory = self.output_dir + '/mpc_c_generated_code'
        self.ocp = ocp

    def setup_first_phase_ocp(self):
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        # formulate acados OCP for 1st phase
        phase_0 = AcadosOcp()
        phase_0.model.f_expl_expr = self.model.x_dot
        phase_0.model.x = self.model.x_sym
        phase_0.model.xdot = cs.MX.sym("xdot", phase_0.model.x.shape)
        phase_0.model.f_impl_expr = phase_0.model.xdot - phase_0.model.f_expl_expr
        phase_0.model.u = self.model.u_sym
        phase_0.model.name = "quadrotor_3D_ph0"
        phase_0.cost.cost_type = 'LINEAR_LS'
        phase_0.cost.W = scipy.linalg.block_diag(self.Q/self.dt, self.R/self.dt)  # scaling by dt as Q, R are given at discrete time
        phase_0.cost.Vx = np.zeros((ny, nx))
        phase_0.cost.Vx[:nx, :nx] = np.eye(nx)
        phase_0.cost.Vu = np.zeros((ny, nu))
        phase_0.cost.Vu[nx:(nx + nu), :nu] = np.eye(nu)
        phase_0.cost.yref = np.zeros((ny, ))
        # constraints 1st phase
        state_constraint_expr_list = []
        input_constraint_expr_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(phase_0.model.x))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(phase_0.model.u))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        phase_0 = self.processing_acados_constraints_expression(phase_0, h0_expr, h_expr, he_expr)
        # slack costs for nonlinear constraints
        if self.soft_constraints:
            # slack variables for all constraints
            phase_0.constraints.Jsh_0 = np.eye(h0_expr.shape[0])
            phase_0.constraints.Jsh = np.eye(h_expr.shape[0])
            # slack penalty
            L2_pen = self.soft_penalty
            L1_pen = self.soft_penalty
            phase_0.cost.Zl_0 = L2_pen * np.ones(h0_expr.shape[0])
            phase_0.cost.Zu_0 = L2_pen * np.ones(h0_expr.shape[0])
            phase_0.cost.zl_0 = L1_pen * np.ones(h0_expr.shape[0])
            phase_0.cost.zu_0 = L1_pen * np.ones(h0_expr.shape[0])
            phase_0.cost.Zu = L2_pen * np.ones(h_expr.shape[0])
            phase_0.cost.Zl = L2_pen * np.ones(h_expr.shape[0])
            phase_0.cost.zl = L1_pen * np.ones(h_expr.shape[0])
            phase_0.cost.zu = L1_pen * np.ones(h_expr.shape[0])
        # placeholder initial state constraint
        x_init = np.zeros((nx))
        phase_0.constraints.x0 = x_init
        return phase_0

    def setup_second_phase_ocp(self):
        nx, nu = self.model.nx, self.model.nu
        ny = nx + nu
        ny_e = nx
        phase_1 = AcadosOcp()
        phase_1.model.x = self.model.x_sym
        phase_1.model.u = self.model.u_sym

        state_constraint_expr_list = []
        input_constraint_expr_list = []
        for sc_i, state_constraint in enumerate(self.state_constraints_sym):
            state_constraint_expr_list.append(state_constraint(phase_1.model.x))
        for ic_i, input_constraint in enumerate(self.input_constraints_sym):
            input_constraint_expr_list.append(input_constraint(phase_1.model.u))

        h_expr_list = state_constraint_expr_list + input_constraint_expr_list
        h_expr = cs.vertcat(*h_expr_list)
        h0_expr = cs.vertcat(*h_expr_list)
        he_expr = cs.vertcat(*state_constraint_expr_list)  # terminal constraints are only state constraints
        # pass the constraints to the ocp object
        phase_1 = self.processing_acados_constraints_expression(phase_1, h0_expr, h_expr, he_expr)
        phase_1.model.name = "quadrotor_3D_ph1"
        phase_1.cost.cost_type = 'CONVEX_OVER_NONLINEAR'
        phase_1.cost.cost_type_e = 'CONVEX_OVER_NONLINEAR'
        cost_W = scipy.linalg.block_diag(self.Q/self.dt, self.R/self.dt)  # scaling by dt as Q, R are given at discrete time
        cost_W_e = self.Q if not self.use_lqr_gain_and_terminal_cost else self.P
        phase_1.model.cost_y_expr_e = phase_1.model.x
        r = cs.MX.sym("r", ny)
        r_e = cs.MX.sym("r_e", ny_e)
        phase_1.model.cost_r_in_psi_expr = r
        phase_1.model.cost_r_in_psi_expr_e = r_e
        phase_1.model.cost_psi_expr = 0.5 * (r.T @ cost_W @ r)
        phase_1.model.cost_psi_expr_e = 0.5 * (r_e.T @ cost_W_e @ r_e)
        phase_1.cost.yref = np.zeros((ny, ))
        phase_1.cost.yref_e = np.zeros((ny_e, ))

        if self.second_phase.method == "clc":
            f_dyn_disc = cs.Function("f_dyn_disc", [self.model.x_sym, self.model.u_sym], [self.model.x_dot])
            pi_param = cs.MX.sym("pi_p", nx)
            phase_1.model.p = pi_param
            phase_1.parameter_values = np.zeros(nx)
            x_next = euler_discrete_casadi_expr(f_dyn_disc, phase_1.model.x, self.pi_casadi_fn(cs.veccat(phase_1.model.x, pi_param)), self.dt)
            phase_1.model.disc_dyn_expr = x_next
            phase_1.model.cost_y_expr = cs.vertcat(phase_1.model.x, self.pi_casadi_fn(cs.veccat(phase_1.model.x, pi_param)).T)
            constr_expr = cs.vertcat(self.pi_casadi_fn(cs.veccat(phase_1.model.x, pi_param)).T, phase_1.model.x)

        elif self.second_phase.method in ["pept", "pt"]:
            phase_1.model.f_expl_expr = self.model.x_dot
            phase_1.model.xdot = cs.MX.sym("xdot", phase_1.model.x.shape)
            phase_1.model.f_impl_expr = phase_1.model.xdot - phase_1.model.f_expl_expr
            phase_1.model.cost_y_expr = cs.vertcat(phase_1.model.x, phase_1.model.u)
            constr_expr = cs.vertcat(phase_1.model.u, phase_1.model.x)

        phase_1 = formulate_constraint_as_log_barrier(phase_1,
            constr_expr,
            float(self.second_phase.barrier_parameter)/self.dt,
            cs.vertcat(phase_1.constraints.ubu, phase_1.constraints.ubx),
            cs.vertcat(phase_1.constraints.lbu, phase_1.constraints.lbx),
            constraint_type='path',
            )
        phase_1 = formulate_constraint_as_log_barrier(phase_1,
            phase_1.model.x,
            float(self.second_phase.barrier_parameter),
            phase_1.constraints.ubx_e,
            phase_1.constraints.lbx_e,
            constraint_type='terminal',
            )
        phase_1.constraints.lbu = np.array([])
        phase_1.constraints.ubu = np.array([])
        phase_1.constraints.idxbu = np.array([])
        phase_1.constraints.lbx = np.array([])
        phase_1.constraints.ubx = np.array([])
        phase_1.constraints.idxbx = np.array([])
        phase_1.constraints.idxbx_e = np.array([])
        phase_1.constraints.lbx_e = np.array([])
        phase_1.constraints.ubx_e = np.array([])

        return phase_1

    def processing_acados_constraints_expression(self,
                                                 ocp: AcadosOcp,
                                                 h0_expr: cs.MX,
                                                 h_expr: cs.MX,
                                                 he_expr: cs.MX,
                                                 ) -> AcadosOcp:
        '''Preprocess the constraints to be compatible with acados.
            Args:
                ocp (AcadosOcp): acados ocp object
                h0_expr (cs.MX expression): initial state constraints
                h_expr (cs.MX expression): state and input constraints
                he_expr (cs.MX expression): terminal state constraints
            Returns:
                ocp (AcadosOcp): acados ocp object with constraints set.

        An alternative way to set the constraints is to use bounded constraints of acados:
        # bounded input constraints
        idxbu = np.where(np.sum(self.env.constraints.input_constraints[0].constraint_filter, axis=0) != 0)[0]
        ocp.constraints.Jbu = np.eye(nu)
        ocp.constraints.lbu = self.env.constraints.input_constraints[0].lower_bounds
        ocp.constraints.ubu = self.env.constraints.input_constraints[0].upper_bounds
        ocp.constraints.idxbu = idxbu # active constraints dimension
        '''
        ub = {}
        lb = {}
        # Check if given constraints are double-sided with heuristics based on the constraint jacobian:
        h_list = [h0_expr, h_expr, he_expr]
        stage_type = ["initial", "path", "terminal"]
        for h_s, s_type in zip(h_list, stage_type):
            jac_h_expr = cs.jacobian(h_s, cs.vertcat(ocp.model.x, ocp.model.u))
            jac_h_fn = cs.Function("jac_h_fn", [ocp.model.x, ocp.model.u], [jac_h_expr])
            jac_eval = jac_h_fn(0, 0).full()

            # Check jacobian by blocks:
            nx = ocp.model.x.shape[0]
            nu = ocp.model.u.shape[0]
            if jac_eval.shape[0] == nx * 2:
                if np.all(jac_eval[:nx, :nx] == -jac_eval[nx:, :nx]):
                    h_double_bounded = True
                    case = 1
            elif jac_eval.shape[0] == nu * 2:
                if np.all(jac_eval[:nu, nx:] == -jac_eval[nu:, nx:]):
                    h_double_bounded = True
                    case = 2
            elif jac_eval.shape[0] == nx * 2 + nu * 2:
                if np.all(jac_eval[:nx, :nx] == -jac_eval[nx:2*nx, :nx]) and \
                    np.all (jac_eval[nx*2:nx*2+nu, nx:] == -jac_eval[nx*2+nu:, nx:]):
                    h_double_bounded = True
                    case = 3
            else:
                h_double_bounded = False

            if h_double_bounded:
                h_fn = cs.Function("h_fn", [ocp.model.x, ocp.model.u], [h_s])
                if all(jac_eval[0, :] >= 0):
                    start_with_ub = True
                else:
                    start_with_ub = False
                if case == 1:
                    if start_with_ub:
                        uh = np.array(-h_fn(0,0)[:nx])
                        lh = np.array(h_fn(0,0)[nx:])
                        h_s_short = h_s[:nx] + uh
                    else:
                        lh = np.array(h_fn(0,0)[:nx])
                        uh = np.array(-h_fn(0,0)[nx:])
                        h_s_short = h_s[:nx] - lh
                elif case == 2:
                    if start_with_ub:
                        uh = np.array(-h_fn(0,0)[:nu])
                        lh = np.array(h_fn(0,0)[nu:])
                        h_s_short = h_s[:nu] + uh
                    else:
                        lh = np.array(h_fn(0,0)[:nu])
                        uh = np.array(-h_fn(0,0)[nu:])
                        h_s_short = h_s[:nu] - lh
                elif case == 3:
                    if start_with_ub:
                        uh = np.concatenate([-h_fn(0,0)[:nx], -h_fn(0,0)[nx*2:nx*2+nu]])
                        lh = np.concatenate([h_fn(0,0)[nx:nx*2], h_fn(0,0)[nx*2+nu:]])
                        h_s_short = cs.vertcat(h_s[:nx], h_s[2*nx:nx*2+nu]) + uh
                    else:
                        lh = np.concatenate([h_fn(0,0)[:nx], h_fn(0,0)[nx*2:nx*2+nu]])
                        uh = np.concatenate([-h_fn(0,0)[nx:nx*2], -h_fn(0,0)[nx*2+nu:]])
                        h_s_short = cs.vertcat(h_s[:nx], h_s[2*nx:nx*2+nu]) - lh
                if s_type == "initial":
                    ocp.model.con_h_expr_0 = -h_s_short
                    ocp.constraints.lh_0 = lh
                    ocp.constraints.uh_0 = uh
                elif s_type == "path":
                    ocp.model.con_h_expr = -h_s_short
                    ocp.constraints.lh = lh
                    ocp.constraints.uh = uh
                elif s_type == "terminal":
                    ocp.model.con_h_expr_e = -h_s_short
                    ocp.constraints.lh_e = lh
                    ocp.constraints.uh_e = uh
                detect_constraint_structure(ocp.model, ocp.constraints, stage_type=s_type)
            else:
                if s_type == "initial":
                    ub.update({'h0': set_acados_constraint_bound(h0_expr, 'ub', self.constraint_tol)})
                    lb.update({'h0': set_acados_constraint_bound(h0_expr, 'lb')})
                elif s_type == "path":
                    ub.update({'h': set_acados_constraint_bound(h_expr, 'ub', self.constraint_tol)})
                    lb.update({'h': set_acados_constraint_bound(h_expr, 'lb')})
                elif s_type == "terminal":
                    ub.update({'he': set_acados_constraint_bound(he_expr, 'ub', self.constraint_tol)})
                    lb.update({'he': set_acados_constraint_bound(he_expr, 'lb')})

        if ub != {}:
            # make sure all the ub and lb are 1D numpy arrays
            # (see: https://discourse.acados.org/t/infeasible-qps-when-using-nonlinear-casadi-constraint-expressions/1595/5?u=mxche)
            for key in ub.keys():
                ub[key] = ub[key].flatten() if ub[key].ndim != 1 else ub[key]
                lb[key] = lb[key].flatten() if lb[key].ndim != 1 else lb[key]
            # check ub and lb dimensions
            for key in ub.keys():
                assert ub[key].ndim == 1, f'ub[{key}] is not 1D numpy array'
                assert lb[key].ndim == 1, f'lb[{key}] is not 1D numpy array'
            assert ub['h'].shape == lb['h'].shape, 'h_ub and h_lb have different shapes'
            # update acados ocp constraints
            for key in ub.keys():
                if key == 'h0':
                    ocp.model.con_h_expr_0 = h0_expr
                    ocp.dims.nh_0 = h0_expr.shape[0]
                    ocp.constraints.uh_0 = ub['h0']
                    ocp.constraints.lh_0 = lb['h0']
                elif key == 'h':
                    ocp.model.con_h_expr = h_expr
                    ocp.dims.nh =  h_expr.shape[0]
                    ocp.constraints.uh = ub['h']
                    ocp.constraints.lh = lb['h']
                elif key == 'he':
                    ocp.model.con_h_expr_e = he_expr
                    ocp.dims.nh_e =  he_expr.shape[0]
                    ocp.constraints.uh_e = ub['he']
                    ocp.constraints.lh_e = lb['he']

        return ocp

    @timing
    def select_action(self,
                      obs,
                      info=None
                      ):
        '''Solves nonlinear mpc problem to get next action.

        Args:
            obs (ndarray): Current state/observation.
            info (dict): Current info

        Returns:
            action (ndarray): Input/action to the task/env.
        '''
        nx, nu = self.model.nx, self.model.nu

        goal_states = self.get_references()
        if self.mode == 'tracking':
            self.traj_step += 1

        if self.second_phase.method == "rl":
            tic = time.time()
            action = self.pi_eval(cs.veccat(obs, goal_states[:, 0]))
            toc = time.time()
            self.results_dict['t_wall'].append([toc-tic, np.nan, np.nan])

        else:
            if self.warmstart:
                if self.x_guess is None or self.u_guess is None:
                    # compute initial guess with the method specified in 'warmstart_type'
                    self.compute_initial_guess(obs)
                    for idx in range(self.T + 1):
                        init_x = self.x_guess[:, idx]
                        self.acados_ocp_solver.set(idx, 'x', init_x)
                    ws_horizon = self.short_horizon if self.second_phase.method == "clc" else self.T
                    for idx in range(ws_horizon):
                        if nu == 1:
                            init_u = np.array([self.u_guess[idx]])
                        else:
                            init_u = self.u_guess[:, idx]
                        self.acados_ocp_solver.set(idx, 'u', init_u)

            # set reference for the control horizon
            y_ref = np.concatenate((goal_states[:, :-1],
                                    np.repeat(self.U_EQ.reshape(-1, 1), self.T, axis=1)), axis=0)

            for idx in range(self.ocp.N_list[0]):
                self.acados_ocp_solver.cost_set(idx, 'yref', y_ref[:, idx])
            for idx in range(self.ocp.N_list[0], self.T):
                self.acados_ocp_solver.cost_set(idx, 'yref', np.concatenate([y_ref[:, idx], np.zeros((32,))]))
            y_ref_e = goal_states[:, -1]
            self.acados_ocp_solver.cost_set(self.T, 'yref', np.concatenate([y_ref_e, np.zeros((24,))]))

            if self.second_phase.method == "clc":
                if self.second_phase.initialization == "warm_starting":
                    for i in range(self.ocp.N_list[0], self.T):
                        self.acados_ocp_solver.set(i, "p", goal_states[:, i+1])
                elif self.second_phase.initialization == "shifting":
                    goal_states = self.get_references()
                    for i in range(self.T):
                        xk_next = self.acados_ocp_solver.get(i+1, "x")
                        self.acados_ocp_solver.set(i, "x", xk_next)
                        if i < self.ocp.N_list[0]-1:
                            uk_next = self.acados_ocp_solver.get(i+1, "u")
                            self.acados_ocp_solver.set(i, "u", uk_next)
                        if i >= self.ocp.N_list[0]:
                            self.acados_ocp_solver.set(i, "p", goal_states[:, i+1])
                    self.acados_ocp_solver.set(self.T, "x", xk_next)

            elif self.second_phase.method == "pept":
                if self.second_phase.initialization_pept == "rollout":
                    if self.second_phase.initialization == "warm_starting":
                        xk = self.acados_ocp_solver.get(self.ocp.N_list[0], "x")
                        for i in range(self.ocp.N_list[0], self.T):
                            uk = self.pi_eval(cs.veccat(xk, goal_states[:, i+1]))
                            xk_next = self.acados_integrator.simulate(xk, uk)
                            self.acados_ocp_solver.set(i, "u", uk)
                            self.acados_ocp_solver.set(i+1, "x", xk_next)
                            xk = xk_next
                    elif self.second_phase.initialization == "shifting":
                        goal_states = self.get_references()
                        for i in range(self.ocp.N_list[0]):
                            xk_next = self.acados_ocp_solver.get(i+1, "x")
                            self.acados_ocp_solver.set(i, "x", xk_next)
                            uk_next = self.acados_ocp_solver.get(i+1, "u")
                            self.acados_ocp_solver.set(i, "u", uk_next)
                        xk = self.acados_ocp_solver.get(self.ocp.N_list[0]+1, "x")
                        self.acados_ocp_solver.set(self.ocp.N_list[0], "x", xk)
                        for i in range(self.ocp.N_list[0], self.T):
                            uk = self.pi_eval(cs.veccat(xk, goal_states[:, i+1]))
                            xk_next = self.acados_integrator.simulate(xk, uk)
                            self.acados_ocp_solver.set(i, "u", uk)
                            self.acados_ocp_solver.set(i+1, "x", xk_next)
                            xk = xk_next

                elif self.second_phase.initialization_pept == "stage_wise":
                    if self.second_phase.initialization == "warm_starting":
                        for i in range(self.ocp.N_list[0], self.T):
                            xk = self.acados_ocp_solver.get(i-1, "x")
                            uk = self.pi_eval(cs.veccat(xk, goal_states[:, i]))
                            xk_next = self.acados_integrator.simulate(xk, uk)
                            self.acados_ocp_solver.set(i, "u", self.pi_eval(cs.veccat(self.acados_ocp_solver.get(i, "x"), goal_states[:, i+1])))
                            self.acados_ocp_solver.set(i, "x", xk_next)
                        xk = self.acados_ocp_solver.get(self.T-1, "x")
                        uk = self.pi_eval(cs.veccat(xk, goal_states[:, self.T]))
                        self.acados_ocp_solver.set(self.T, "x", self.acados_integrator.simulate(xk, uk))
                    elif self.second_phase.initialization == "shifting":
                        goal_states = self.get_references()
                        for i in range(self.ocp.N_list[0]):
                            xk_next = self.acados_ocp_solver.get(i+1, "x")
                            self.acados_ocp_solver.set(i, "x", xk_next)
                            uk_next = self.acados_ocp_solver.get(i+1, "u")
                            self.acados_ocp_solver.set(i, "u", uk_next)
                        for i in range(self.ocp.N_list[0], self.T):
                            xk = self.acados_ocp_solver.get(i, "x")
                            uk = self.pi_eval(cs.veccat(xk, goal_states[:, i]))
                            xk_next = self.acados_integrator.simulate(xk, uk)
                            self.acados_ocp_solver.set(i, "u", self.pi_eval(cs.veccat(self.acados_ocp_solver.get(i+1, "x"), goal_states[:, i+1])))
                            self.acados_ocp_solver.set(i, "x", xk_next)
                        xk = self.acados_ocp_solver.get(self.T, "x")
                        uk = self.pi_eval(cs.veccat(xk, goal_states[:, self.T]))
                        self.acados_ocp_solver.set(self.T, "x", self.acados_integrator.simulate(xk, uk))

            elif self.second_phase.method == "pt":
                if self.second_phase.initialization == "warm_starting":
                    pass
                elif self.second_phase.initialization == "shifting":
                    for i in range(self.T):
                        xk_next = self.acados_ocp_solver.get(i+1, "x")
                        self.acados_ocp_solver.set(i, "x", xk_next)
                        if i < self.T-1:
                            uk_next = self.acados_ocp_solver.get(i+1, "u")
                            # Else: repeat last control
                        self.acados_ocp_solver.set(i, "u", uk_next)
                    self.acados_ocp_solver.set(self.T, "x", xk_next)

            if self.use_RTI:
                # solve the optimization problem
                self.acados_ocp_solver.options_set('rti_phase', 1)
                status = self.acados_ocp_solver.solve()
                time_preparation = self.acados_ocp_solver.get_stats("time_preparation")
                # set initial condition (0-th state)
                self.acados_ocp_solver.set(0, 'lbx', obs)
                self.acados_ocp_solver.set(0, 'ubx', obs)
                self.acados_ocp_solver.options_set('rti_phase', 2)
                # solve
                status = self.acados_ocp_solver.solve()
                time_feedback = self.acados_ocp_solver.get_stats("time_feedback")
                self.results_dict['t_wall'].append([time_preparation + time_feedback, time_preparation, time_feedback])
            else:
                # set initial condition (0-th state)
                self.acados_ocp_solver.set(0, 'lbx', obs)
                self.acados_ocp_solver.set(0, 'ubx', obs)
                # solve the optimization problem
                status = self.acados_ocp_solver.solve()
                time_tot = self.acados_ocp_solver.get_stats('time_tot')
                self.results_dict['t_wall'].append([time_tot, np.nan, np.nan])
            if status not in [0, 2]:
                self.acados_ocp_solver.print_statistics()
                print(colored(f'acados returned status {status}. Terminating the simulation.', 'red'))
                action = np.nan * np.ones(nu)
                # action = self.acados_ocp_solver.get(0, 'u')
            elif status == 2:
                self.acados_ocp_solver.print_statistics()
                print(f'acados returned status {status}. ')
                action = self.acados_ocp_solver.get(0, 'u')
            else:
                action = self.acados_ocp_solver.get(0, 'u')

            # get the open-loop solution
            if self.x_prev is None and self.u_prev is None:
                self.x_prev = np.zeros((nx, self.T + 1))
                self.u_prev = np.zeros((nu, self.T))
            if self.u_prev is not None and nu == 1:
                self.u_prev = self.u_prev.reshape((1, -1))
            for i in range(self.T + 1):
                self.x_prev[:, i] = self.acados_ocp_solver.get(i, 'x')
            for i in range(self.T):
                self.u_prev[:, i] = self.acados_ocp_solver.get(i, 'u')
            if nu == 1:
                self.u_prev = self.u_prev.flatten()

            self.x_guess = self.x_prev
            self.u_guess = self.u_prev
            self.results_dict['horizon_states'].append(deepcopy(self.x_prev))
            self.results_dict['horizon_inputs'].append(deepcopy(self.u_prev))
            self.results_dict['goal_states'].append(deepcopy(goal_states))

        self.prev_action = action
        if self.use_lqr_gain_and_terminal_cost:
            action += self.lqr_gain @ (obs - self.x_prev[:, 0])

        return action
