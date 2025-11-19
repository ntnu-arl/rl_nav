import time
from collections import deque
from typing import Dict, Tuple

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import *


from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from gymnasium import spaces

from torch import nn

def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument(
        "--env_agents",
        default=-1,
        type=int,
        help="Num agents in each env (default: -1, means use default value from isaacgymenvs env yaml config file)",
    )
    p.add_argument(
        "--obs_key",
        default="obs",
        type=str,
        help='IsaacGym envs return dicts, some envs return just "obs", and some return "obs" and "states".'
        "States key denotes the full state of the environment, and obs key corresponds to limited observations "
        'available in real world deployment. If we use "states" here we can train will full information '
        "(although the original idea was to use asymmetric training - critic sees full state and policy only sees obs).",
    )
    p.add_argument(
        "--subtask",
        default=None,
        type=str,
        help="Subtask for envs that support it (i.e. AllegroKuka regrasping or manipulation or throw).",
    )
    p.add_argument(
        "--ige_api_version",
        default="preview4",
        type=str,
        choices=["preview3", "preview4"],
        help="We can switch between different versions of IsaacGymEnvs API using this parameter.",
    )
    p.add_argument(
        "--eval_stats",
        default=False,
        type=str2bool,
        help="Whether to collect env stats during evaluation.",
    )

def override_default_params_func(env, parser):
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=10000000,
        use_rnn=False,
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.1,
        rollout=24,
        max_grad_norm=0.0,
        batch_size=2048,
        num_batches_per_epoch=2,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=True,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        save_best_after=int(1e5),
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=True,
        use_env_info_cache=False,  # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="overwrite",
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    position_setpoint_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        gamma=0.99,
        rollout=16,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=16384,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    navigation_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=2048,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    lidar_navigation_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=128,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=1024,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.001,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
        train_dir="../models",
        load_checkpoint_kind="best",
    ),

)

def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


class NN_Inference_ROS(nn.Module):
    def __init__(self, cfg: Config, env_cfg) -> None:
        super().__init__()
        self.cfg = load_from_checkpoint(cfg)
        print("cfg: ", self.cfg)
        self.cfg.num_envs = 1
        self.num_actions = 4
        self.num_obs = 17 + 16*20 #15 #+ self.num_actions * 10
        self.num_agents = 1
        self.observation_space = spaces.Dict(dict(observations=convert_space(spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf))))
        self.action_space = convert_space(spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.))
        self.init_env_info()
        self.actor_critic = create_actor_critic(self.cfg, self.observation_space, self.action_space)
        self.actor_critic.eval()
        device = torch.device("cpu")#"cuda:0" if self.cfg.device == "cpu" else "cuda")
        self.actor_critic.model_to_device(device)
        print("Model:\n\n", self.actor_critic)
        # Load policy into model
        policy_id = 0 #self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*"
        )
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.rnn_states = torch.zeros(
            [self.num_agents, get_rnn_size(self.cfg)],
            dtype=torch.float32,
            device=device,
        )

    def init_env_info(self):
        self.env_info = EnvInfo(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_agents=self.num_agents,
            gpu_actions=self.cfg.env_gpu_actions,
            gpu_observations=self.cfg.env_gpu_observations,
            action_splits=None,
            all_discrete=None,
            frameskip=self.cfg.env_frameskip,
        )

    def reset(self):
        self.rnn_states[:] = 0.0

    def get_action(self, obs):
        start_time = time.time()
        with torch.no_grad():
            # put obs to device
            processed_obs = prepare_and_normalize_obs(self.actor_critic, obs)
            policy_outputs = self.actor_critic(processed_obs, self.rnn_states)
            # sample actions from the distribution by default
            actions = policy_outputs["actions"]
            action_distribution = self.actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)

            self.rnn_states = policy_outputs["new_rnn_states"]
        #actions_np = actions[0].cpu().numpy()
        #print("Time to get action:", time.time() - start_time)
        return actions

from sample_factory.model.encoder import *

class CustomEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # the observatiosn are in the following format:
        # 17 dim state + actions
        # 80 dim lidar readings
        
        # encode lidar readings using conv network and then combine with state vector and pass through MLP
        # all are flattened into "observations" key
        self.encoders = nn.ModuleDict()

        state_action_input_size = 17
        lidar_input_size = 16*20
        out_size = 0
        out_size_cnn = 0
        out_size += obs_space["observations"].shape[0] - lidar_input_size

        # self.encoders["obs_image"] = make_img_encoder(cfg, spaces.Box(low=-1, high=1.5, shape=(1, 16, 20)))
        # out_size += self.encoders["obs_image"].get_out_size()
        ###
        # input is 16 x 20 dims lidar readings
        self.encoders["obs_image"] = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, 16, 20)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (B, 16, 8, 10)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (B, 32, 8, 10)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (B, 32, 4, 5)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 4, 5)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 2, 2)
            nn.Flatten(),  # (B, 64*2*2)
        )
        out_size_cnn += 128
        out_size += out_size_cnn
        ###

        self.encoder_out_size = out_size
        mlp_layers = [256, 128, 64]
        mlp_input_size = out_size
        mlp = []
        for layer_size in mlp_layers:
            mlp.append(nn.Linear(mlp_input_size, layer_size))
            mlp.append(nn.ELU())
            mlp_input_size = layer_size
        self.mlp_head_custom = nn.Sequential(*mlp)
        if len(mlp_layers) > 0:
            self.encoder_out_size = mlp_layers[-1]
        else:
            self.encoder_out_size = out_size
            self.mlp_head_custom = nn.Identity()


    def forward(self, obs_dict):
        x_state_action = obs_dict["observations"][:, :17]
        x_lidar = obs_dict["observations"][:, 17:].unsqueeze(1).view(-1, 1, 16, 20)  # (B, 1, 8, 10)
        x_lidar_encoding = self.encoders["obs_image"](x_lidar)
        x = torch.cat([x_state_action, x_lidar_encoding], dim=-1)
        x = self.mlp_head_custom(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


from sample_factory.algo.utils.context import global_model_factory

def make_custom_encoder(cfg, obs_space):
    return CustomEncoder(cfg, obs_space)

def register_aerialgym_custom_components():
    global_model_factory().register_encoder_factory(CustomEncoder)

def SF_model_initializer():
    """Script entry point."""
    print("Starting inference script")
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    model = NN_Inference_ROS(cfg, None)
    obs = {
        "observations": torch.zeros((1, model.num_obs), dtype=torch.float32)
    }
    action = model.get_action(obs)
    print("Sample action from the model:", action)
    return model


if __name__ == "__main__":
    model = SF_model_initializer()
    # measure GPU usage from this script

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size (number of parameters): {model_size}")
    import pynvml
    pynvml.nvmlInit()
    model.eval()
    model_input = torch.zeros((1, model.num_obs), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(1000):
            action = model.get_action({"observations": model_input})
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            action = model.get_action({"observations": model_input})
    torch.cuda.synchronize()
    print("Action:", action)
    print("Time to get action:", (time.time() - start_time)/1000.0)


    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  #
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory total: {info.total / 1024 ** 2} MB")
    print(f"GPU memory used: {info.used / 1024 ** 2} MB")
    print(f"GPU memory free: {info.free / 1024 ** 2} MB")
    # exit(status.status.value)