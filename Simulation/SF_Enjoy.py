import time
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from copy import deepcopy

import argparse, sys
sys.path.append("../sample-factory")

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log
from sample_factory.envs.env_utils import register_env
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

from point_trajectory_env import PointTrajectoryEnv

def enjoy_point_trajectory():
    # parse and strip custom flags (record_video, render_mode, trajectory)
    _custom_parser = argparse.ArgumentParser(add_help=False)
    _custom_parser.add_argument('--record_video', action='store_true',
                                help='Record the enjoy rendering to a video file')
    _custom_parser.add_argument('--render_mode', type=str, default=None,
                                help='Override render mode for enjoy')
    _custom_parser.add_argument('--trajectory', type=str,
                                choices=['circle', 'square', 'waypoint', 'waypoints', 'hold', 'random'],
                                help='Preset target trajectory')
    _custom_parser.add_argument('--hold_point', type=str, default=None,
                                help='Hold point as "x,y" for waypoint/hold mode')
    _custom_args, _remaining = _custom_parser.parse_known_args()

    # rebuild sys.argv without our custom flags so SF parser doesn't choke
    _new_argv = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        # strip our custom flags
        if arg in ('--record_video', '--save_video'):
            continue
        if arg.startswith('--render_mode'):
            if '=' not in arg:
                skip_next = True
            continue
        if arg == '--trajectory' or arg == '--hold_point':
            skip_next = True
            continue
        if arg.startswith('--trajectory=') or arg.startswith('--hold_point='):
            continue
        _new_argv.append(arg)
    sys.argv = _new_argv

def make_point_env(full_env_name: str, cfg=None, env_config=None, render_mode=None):
    merged = dict(env_config or {})
    # pass through trajectory knobs (use existing names you support)
    for k in [
        "trajectory",
        "circle_radius", "circle_speed",
        "square_side", "square_speed",
        "dt", "tau_v", "tau_w",
        "v_f_max", "v_r_max", "w_max",
        "tgt_ou_theta", "tgt_ou_sigma", "tgt_v_max",
        "theta_gate_r0", "theta_gate_r1", "theta_gate_min",
        "capture_radius", "capture_bonus",
        "world_radius", "fpv_fov", "fpv_rmax",
        "max_steps", "heading",
        "auto_reset_on_timeout",
    ]:
        if hasattr(cfg, k):
            merged[k] = getattr(cfg, k)
    # parse hold_point "x,y" string to list if needed
    hp = merged.get("hold_point", None)
    if isinstance(hp, str):
        try:
            xs, ys = hp.split(",")
            merged["hold_point"] = [float(xs), float(ys)]
        except Exception:
            merged["hold_point"] = [0.0, 0.0]
    return PointTrajectoryEnv(full_env_name, cfg=cfg, env_config=merged, render_mode=render_mode)

def register_point_env():
    register_env("point_trajectory", make_point_env)
    
def add_point_env_args(env, parser: argparse.ArgumentParser, evaluation=False):
    parser.add_argument("--trajectory",
                        default="random",
                        type=str,
                        choices=["circle", "square", "random", "waypoints", "waypoint", "waypoint2", "hold"],
                        help="Target motion mode. 'waypoint/hold' = single fixed point, 'waypoints' = polyline.")

    # New, useful environment knobs
    g = parser.add_argument_group("PointTrajectoryEnv")
    g.add_argument("--dt", type=float, default=0.1)
    g.add_argument("--max_steps", type=int, default=100)
    g.add_argument("--world_radius", type=float, default=25.0)
    g.add_argument("--hold_point", type=str, default=None,
                   help='Hold point "x,y" for trajectory=waypoint/hold')

    # dynamics
    g.add_argument("--v_f_max", type=float, default=2.5)
    g.add_argument("--v_r_max", type=float, default=2.5)
    g.add_argument("--w_max", type=float, default=2.0)
    g.add_argument("--tau_v", type=float, default=0.4)
    g.add_argument("--tau_w", type=float, default=0.4)

    # target OU
    g.add_argument("--tgt_ou_theta", type=float, default=0.7)
    g.add_argument("--tgt_ou_sigma", type=float, default=1.0)
    g.add_argument("--tgt_v_max", type=float, default=2.0)

    # rewards & gates
    g.add_argument("--w_r", type=float, default=1.0)
    g.add_argument("--w_theta", type=float, default=0.25)
    g.add_argument("--w_act", type=float, default=0.01)
    g.add_argument("--theta_gate_r0", type=float, default=1.0)
    g.add_argument("--theta_gate_r1", type=float, default=5.0)
    g.add_argument("--theta_gate_min", type=float, default=0.1)
    g.add_argument("--capture_radius", type=float, default=0.5)
    g.add_argument("--capture_bonus", type=float, default=5.0)

def point_trajectory_override_defaults(env, parser: argparse.ArgumentParser):
    parser.set_defaults(
        encoder_mlp_layers=[64, 64, 64],
        learning_rate=5e-4,
        train_for_env_steps=1_000_000,
        env_frameskip=1,
    )

def parse_args(argv=None, evaluation=False):
    argv = ['--env=point_trajectory', '--experiment=Tan05', '--train_dir=./train_dir', '--load_checkpoint_kind', 'best', '--eval_deterministic=True', '--video_frames=300', '--device=cpu']
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_point_env_args(partial_cfg.env, parser, evaluation=evaluation)
    point_trajectory_override_defaults(partial_cfg.env, parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg

def SF_Enjoy_main():
    """Script entry-point for evaluation with automatic video naming and preset trajectories."""

    register_point_env()
    cfg = parse_args(evaluation=True)
    # satisfy SF enjoy argument requirements
    cfg.cli_args = vars(cfg)
    hxs = torch.zeros(1, 1, 512)[0]
    return SF_Enjoy(cfg), hxs


class SF_Enjoy:
    def __init__(self, cfg: Config):
        self.cfg = deepcopy(cfg)

        self.cfg = load_from_checkpoint(cfg)

        eval_env_frameskip: int = self.cfg.env_frameskip if self.cfg.eval_env_frameskip is None else self.cfg.eval_env_frameskip
        assert (
            self.cfg.env_frameskip % eval_env_frameskip == 0
        ), f"{self.cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
        render_action_repeat: int = self.cfg.env_frameskip // eval_env_frameskip
        self.cfg.env_frameskip = self.cfg.eval_env_frameskip = eval_env_frameskip
        log.debug(f"Using frameskip {self.cfg.env_frameskip} and {render_action_repeat=} for evaluation")

        self.cfg.num_envs = 1

        render_mode = "human"
        env = make_env_func_batched(self.cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode)
        self.env_info = extract_env_info(env, self.cfg)

        if hasattr(env.unwrapped, "reset_on_init"):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False
        
        self.actor_critic = create_actor_critic(self.cfg, env.observation_space, env.action_space)
        self.actor_critic.eval()

        device = torch.device("cpu" if self.cfg.device == "cpu" else "cuda")
        self.actor_critic.model_to_device(device)

        policy_id = self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])

    def max_frames_reached(self,frames):
        return self.cfg.max_num_frames is not None and frames > self.cfg.max_num_frames
    
    def enjoy(self, obs, rnn_states) -> Tuple[StatusCode, float]:
 
        with torch.no_grad():
            normalized_obs = prepare_and_normalize_obs(self.actor_critic, {"obs": obs})

            policy_outputs = self.actor_critic(normalized_obs, rnn_states, action_mask=None)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if self.cfg.eval_deterministic:
                action_distribution = self.actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)

            rnn_states_out = policy_outputs["new_rnn_states"]
            action_mean = policy_outputs["action_logits"][0][0:3]
            action_logstd = policy_outputs["action_logits"][0][3:6]
        return actions, rnn_states_out, action_mean, action_logstd