# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from soundspaces.tasks.action_space import MoveOnlySpaceConfiguration
from soundspaces.simulator import SoundSpacesSim
from soundspaces.datasets.audionav_dataset import AudioNavDataset
from soundspaces.datasets.semantic_audionav_dataset import SemanticAudioNavDataset
from soundspaces.tasks.audionav_task import AudioNavigationTask
from soundspaces.tasks.semantic_audionav_task import SemanticAudioNavigationTask
from soundspaces.tasks.nav import AudioGoalSensor
from soundspaces.tasks.nav import SpectrogramSensor
from soundspaces.tasks.nav import Collision
from soundspaces.challenge import Challenge
from soundspaces.benchmark import Benchmark


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type, Union
import logging
import copy
import random

import numpy as np
import torch

import habitat
from habitat import Config, Env, RLEnv, VectorEnv
from habitat.datasets import make_dataset

from env.habitat import Exploration_Env
from env.config.default import get_config
# from ss_baselines.common.sync_vector_env import SyncVectorEnv

REPLICA_SCENES = ['apartment_0', 'apartment_1', 'apartment_2', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2',
          'frl_apartment_3', 'frl_apartment_4', 'frl_apartment_5', 'office_0', 'office_1', 'office_2',
          'office_3', 'office_4', 'hotel_0', 'room_0', 'room_1', 'room_2']


def construct_envs(args) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created
        auto_reset_done: automatically reset environments when done
    Returns:
        VectorEnv object created according to specification.
    """
    auto_reset_done=True
    config = get_config(args.exp_config, args.opts, args.model_dir, args.run_type, args.overwrite)
    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [Exploration_Env for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    # rearrange scenes in the order of scene size since there is a severe imbalance of data size
    if "replica" in config.TASK_CONFIG.DATASET.SCENES_DIR:
        scenes_new = list()
        for scene in REPLICA_SCENES:
            if scene in scenes:
                scenes_new.append(scene)
        scenes = scenes_new

    if len(scenes) > 0:
        # random.shuffle(scenes)
        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        task_config = config.TASK_CONFIG.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]
            logging.debug('All scenes: {}'.format(','.join(scene_splits[i])))

        # overwrite the task config with top-level config file
        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )
        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS
        task_config.freeze()

        config.defrost()
        config.TASK_CONFIG = task_config
        config.freeze()
        configs.append(config.clone())

    # # use VectorEnv for the best performance and ThreadedVectorEnv for debugging
    # if config.USE_SYNC_VECENV:
    #     env_launcher = SyncVectorEnv
    #     logging.info('Using SyncVectorEnv')
    # elif config.USE_VECENV:
    env_launcher = habitat.VectorEnv
    logging.info('Using VectorEnv')
    # else:
    #     env_launcher = habitat.ThreadedVectorEnv
    #     logging.info('Using ThreadedVectorEnv')

    envs = env_launcher(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(configs, range(num_processes)))),
        auto_reset_done=auto_reset_done
    )
    return envs


def make_env_fn(
    config: Config, rank: int
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.
    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).
    Returns:
        env object created according to specification.
    """
    if not config.USE_SYNC_VECENV:
        level = logging.DEBUG if config.DEBUG else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        random.seed(rank)
        np.random.seed(rank)
        torch.manual_seed(rank)

    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = Exploration_Env(config=config, dataset=dataset)
    env.seed(rank)
    return env
