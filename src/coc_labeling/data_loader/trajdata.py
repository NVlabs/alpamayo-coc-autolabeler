# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig
from trajdata.data_structures.agent import AgentType

from coc_labeling.request import project_speed_to_heading

np.set_printoptions(suppress=True, precision=10, threshold=np.inf)
torch.set_printoptions(sci_mode=False, precision=10, threshold=float("inf"))


def convert_trajdata2data_dict(scene_batch: Any, vector_config: DictConfig) -> Dict[str, Any]:
    """Convert one trajdata scene batch into the internal prompt data dictionary.

    Args:
        scene_batch: trajdata scene batch containing ego/agent histories and futures.
        vector_config: Vector sampling/horizon configuration.

    Returns:
        Dictionary containing ego state features, object trajectories, and metadata.
    """
    # convert the data format to support the agentformer
    data_dict = {}

    # process only one scene_batch currently
    scene_ts = scene_batch.scene_ts.numpy()[0]
    agent_cur = scene_batch.agent_hist[0]
    agent_cur_extent = scene_batch.agent_hist_extent[0]
    agent_fut = scene_batch.agent_fut[0]  # num_agents x num_fut x 6
    agent_fut_extent = scene_batch.agent_fut_extent[0]
    all_agents = scene_batch.agent_names[0]
    agent_type = scene_batch.agent_type[0]
    data_dict["ts"] = scene_ts

    # convert the format to numpy as in the data_dict requirement
    agent_type = agent_type.numpy()
    agent_cur_traj = agent_cur.as_format("x,y,z,h")  # num_agents x (num_his+1) x 4
    agent_cur_extent = agent_cur_extent.numpy()  # num_agents x (num_his+1) x 3, lwh
    agent_cur_speed = agent_cur.as_format("xd,yd")  # num_agents x (num_his+1) x 2
    agent_cur_acce = agent_cur.as_format("xdd,ydd").numpy()  # num_agents x (num_his+1) x 2
    agent_fut_traj = agent_fut.as_format("x,y,z,h").numpy()  # num_agents x num_fut x 4
    agent_fut_extent = agent_fut_extent.numpy()  # num_agents x num_fut x 3
    agent_fut_speed = agent_fut.as_format("xd,yd")  # num_agents x (num_his+1) x 2
    agent_fut_acce = agent_fut.as_format("xdd,ydd").numpy()  # num_agents x (num_his+1) x 2
    # agent_cur timestamp, agent_fut timestamp are both ordered by time
    # agent_cur x -> front, y -> left, z -> up

    agent_cur_traj = agent_cur_traj.numpy()
    agent_cur_speed = agent_cur_speed.numpy()
    agent_fut_speed = agent_fut_speed.numpy()

    # search for the ego index
    ego_index = -1
    agent_name: str
    for index, agent_name in enumerate(all_agents):
        if agent_name == "ego":
            ego_index = index
            break

    # ego agent exists, transformed to the nuScenes coordinate with axes:
    # x -> right, y -> front, z -> up
    if ego_index != -1:
        ########## history trajectories
        # only take history 2s at 2Hz, out of the 10Hz data
        ego_hist_traj = copy.deepcopy(agent_cur_traj[ego_index, :: vector_config.sample_rate, :2])
        ego_hist_traj = ego_hist_traj[:, [1, 0]]  # swap the xy axes
        ego_hist_traj[:, 0] = -ego_hist_traj[:, 0]
        data_dict["ego_hist_traj"] = ego_hist_traj

        ########## future trajectories
        start_index = vector_config.sample_rate - 1  # 0-indexed
        ego_fut_traj = copy.deepcopy(
            agent_fut_traj[
                ego_index,
                start_index : vector_config.fut_length_frame : vector_config.sample_rate,
                :2,
            ]
        )
        ego_fut_traj = ego_fut_traj[:, [1, 0]]  # swap the xy axes
        ego_fut_traj[:, 0] = -ego_fut_traj[:, 0]
        data_dict["ego_fut_traj"] = ego_fut_traj

        ########## velocity
        ego_hist_vel = copy.deepcopy(agent_cur_speed[ego_index])
        ego_hist_vel = ego_hist_vel[:, [1, 0]]  # swap the xy axes
        ego_hist_vel[:, 0] = -ego_hist_vel[:, 0]  # (num_his+1) x 2

        # absolute speed, not vector
        vx, vy = ego_hist_vel[-1]
        v0_absolute = math.sqrt(vx**2 + vy**2)
        ego_hist_vel = ego_hist_vel[:: vector_config.sample_rate]
        data_dict["ego_hist_vel"] = ego_hist_vel

        # future velocity
        ego_fut_vel = copy.deepcopy(agent_fut_speed[ego_index])
        ego_fut_vel = ego_fut_vel[:, [1, 0]]  # swap the xy axes
        ego_fut_vel[:, 0] = -ego_fut_vel[:, 0]  # (num_his+1) x 2
        ego_fut_vel = ego_fut_vel[
            start_index : vector_config.fut_length_frame : vector_config.sample_rate
        ]
        data_dict["ego_fut_vel"] = ego_fut_vel

        ########## heading and angular velocity
        # only take the yaw speed at 2Hz
        ego_hist_heading = agent_cur_traj[ego_index, :, -1]
        ego_fut_heading = agent_fut_traj[ego_index, :, -1]
        # convert from x-axis positive to y-axis positive
        ego_hist_heading = (ego_hist_heading + np.pi / 2) % (2 * np.pi)  # rad
        ego_fut_heading = (ego_fut_heading + np.pi / 2) % (2 * np.pi)  # rad
        ego_heading_hist_fut = np.concatenate((ego_hist_heading, ego_fut_heading), axis=0)
        # Unwrap for derivative computations to avoid artificial +/-2pi jumps.
        ego_heading_hist_fut_unwrapped = np.unwrap(ego_heading_hist_fut)
        angular_vel = np.diff(ego_heading_hist_fut_unwrapped, axis=0) * vector_config.fps
        angular_vel_sample = angular_vel[:: vector_config.sample_rate]
        angular_vel_sample = np.concatenate(
            (angular_vel_sample, np.array([angular_vel[-1]])), axis=0
        )

        # +1 to include the current ts=0
        data_dict["ego_hist_angular_vel"] = angular_vel_sample[
            : vector_config.hist_length_frame_sample + 1
        ]
        data_dict["ego_fut_angular_vel"] = angular_vel_sample[
            vector_config.hist_length_frame_sample + 1 :
        ]
        hist_len = ego_hist_heading.shape[0]
        v_yaw = (
            ego_heading_hist_fut_unwrapped[hist_len - 1]
            - ego_heading_hist_fut_unwrapped[hist_len - 2]
        ) * vector_config.fps  # rad/s
        ego_hist_heading = ego_hist_heading[:: vector_config.sample_rate]
        data_dict["ego_hist_heading"] = ego_hist_heading
        ego_fut_heading = ego_fut_heading[
            start_index : vector_config.fut_length_frame : vector_config.sample_rate
        ]
        data_dict["ego_fut_heading"] = ego_fut_heading

        ########## longitudinal and lateral speed (with projection)
        ego_hist_longitudinal_speed, ego_hist_lateral_speed = project_speed_to_heading(
            ego_hist_vel, ego_hist_heading
        )
        ego_fut_longitudinal_speed, ego_fut_lateral_speed = project_speed_to_heading(
            ego_fut_vel, ego_fut_heading
        )
        data_dict["ego_hist_longitudinal_speed"] = ego_hist_longitudinal_speed
        data_dict["ego_hist_lateral_speed"] = ego_hist_lateral_speed
        data_dict["ego_fut_longitudinal_speed"] = ego_fut_longitudinal_speed
        data_dict["ego_fut_lateral_speed"] = ego_fut_lateral_speed

        ########## acceleration
        ego_hist_acc = copy.deepcopy(agent_cur_acce[ego_index])
        ego_hist_acc = ego_hist_acc[:, [1, 0]]  # swap the xy axes
        ego_hist_acc[:, 0] = -ego_hist_acc[:, 0]  # (num_his+1) x 2
        ax, ay = ego_hist_acc[-1]
        ego_hist_acc = ego_hist_acc[:: vector_config.sample_rate]
        data_dict["ego_hist_acc"] = ego_hist_acc

        # future
        ego_fut_acc = copy.deepcopy(agent_fut_acce[ego_index])
        ego_fut_acc = ego_fut_acc[:, [1, 0]]  # swap the xy axes
        ego_fut_acc[:, 0] = -ego_fut_acc[:, 0]  # (num_his+1) x 2
        ego_fut_acc = ego_fut_acc[
            start_index : vector_config.fut_length_frame : vector_config.sample_rate
        ]
        data_dict["ego_fut_acc"] = ego_fut_acc

        # size
        ego_length, ego_width, _ = agent_cur_extent[ego_index, -1]
        ego_states = [vx, vy, ax, ay, v_yaw, ego_length, ego_width, v0_absolute]
    else:
        raise ValueError("ego agent does not exist")

    # loop through all agents
    objects: List[dict[str, Any]] = []
    agent_id = 0
    for index, agent_name in enumerate(all_agents):
        # skip ego for detection
        if agent_name == "ego":
            ego_index = index
            continue

        # state in the current frame
        x_front, y_left, z_up, heading = agent_cur_traj[index, -1]
        length, width, height = agent_cur_extent[index, -1]

        # meta data
        agent_type_int: int = int(agent_type[index])
        agent_type_str: str = str(AgentType(agent_type_int).name)
        agent_id_global: int = int(agent_name.split("_")[-1])

        # future trajectories,
        # only take the xy position, swap the axis and revert the direction
        start_index = vector_config.sample_rate - 1  # 0-indexed
        traj = copy.deepcopy(
            agent_fut_traj[
                index,
                start_index : vector_config.fut_length_frame : vector_config.sample_rate,
                :2,
            ]
        )
        traj = traj[:, [1, 0]]
        traj[:, 0] = -traj[:, 0]

        # past trajectories
        hist_traj = copy.deepcopy(agent_cur_traj[index, :: vector_config.sample_rate, :2])
        hist_traj = hist_traj[:, [1, 0]]  # swap the xy axes
        hist_traj[:, 0] = -hist_traj[:, 0]

        obj_tmp = {
            "id": agent_id,
            "id_global": agent_id_global,
            "name": agent_type_str,
            "traj": traj,
            # converting bbox
            # x,y,z,dx=w,dy=l,dz=h,rotation_z=heading,rotation_y,rotation_x
            # x -> right
            # y -> front
            # z -> up
            # assuming rotation_y,rotation_x are 0s
            "bbox": np.array([-y_left, x_front, z_up, width, length, height, heading, 0, 0]),
            "traj_past": hist_traj,
        }
        objects.append(obj_tmp)

        # this id is different from the index, since it excludes the ego
        agent_id += 1

    data_dict["objects"] = objects
    data_dict["ego_states"] = ego_states
    data_dict["vector_map"] = (
        None if getattr(scene_batch, "vector_maps", None) is None else scene_batch.vector_maps[0]
    )
    data_dict["world_from_agent_tf"] = scene_batch.centered_world_from_agent_tf
    data_dict["agent_from_world_tf"] = scene_batch.centered_agent_from_world_tf

    return data_dict
