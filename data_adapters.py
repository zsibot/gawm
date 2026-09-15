"""Dataset-specific state targets for LeWM training.

Each adapter reads the schema written in its data config and constructs the
``state_align`` target for one task.  The common pipeline then applies exactly
three transforms: image preprocessing, state-target construction, and action
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import stable_pretraining as spt

from utils import (
    get_column_normalizer,
    get_cube_state_alignment_transform,
    get_img_preprocessor,
    get_pusht_state_alignment_transform,
    get_pingpong_state_alignment_transform,
    get_reacher_state_alignment_transform,
    get_tworoom_state_alignment_transform,
)


@dataclass(frozen=True)
class StateAlignmentSpec:
    target_transform: Callable
    output_dim: int


@dataclass(frozen=True)
class DataPipelineSpec:
    transform: Callable
    action_dim: int
    state_alignment: StateAlignmentSpec


class DatasetAdapter:
    """Base class shared by the five explicit task adapters."""

    name = "base"

    def __init__(self, dataset, config, keys_to_load: Sequence[str]):
        self.dataset = dataset
        self.config = config
        self.keys_to_load = tuple(keys_to_load)

    def require_columns(self, *columns: str) -> None:
        missing = [column for column in columns if column not in self.keys_to_load]
        if missing:
            raise ValueError(
                f"{self.name} adapter requires {missing} in "
                "data.dataset.keys_to_load"
            )

    def build_state_alignment(self) -> StateAlignmentSpec:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Cube
# ---------------------------------------------------------------------------


class CubeAdapter(DatasetAdapter):
    name = "cube"

    def build_state_alignment(self) -> StateAlignmentSpec:
        observation_source = "observation"
        qvel_source = "qvel"
        num_cubes = int(self.config["num_cubes"])
        output_dim = 12 + 9 * num_cubes + 6 + 6 * num_cubes

        self.require_columns(observation_source, qvel_source)

        transform = get_cube_state_alignment_transform(
            self.dataset,
            source=observation_source,
            qvel_source=qvel_source,
            num_cubes=num_cubes,
        )
        return StateAlignmentSpec(
            target_transform=transform,
            output_dim=output_dim,
        )


# ---------------------------------------------------------------------------
# Reacher
# ---------------------------------------------------------------------------


class ReacherAdapter(DatasetAdapter):
    name = "reacher"

    def build_state_alignment(self) -> StateAlignmentSpec:
        observation_source = "observation"

        self.require_columns(observation_source)
        transform = get_reacher_state_alignment_transform(
            self.dataset,
            source=observation_source,
        )
        return StateAlignmentSpec(
            target_transform=transform,
            output_dim=8,
        )


# ---------------------------------------------------------------------------
# PushT
# ---------------------------------------------------------------------------


class PushTAdapter(DatasetAdapter):
    name = "pusht"

    def build_state_alignment(self) -> StateAlignmentSpec:
        state_source = "state"

        self.require_columns(state_source)
        transform = get_pusht_state_alignment_transform(
            self.dataset,
            source=state_source,
        )
        return StateAlignmentSpec(
            target_transform=transform,
            output_dim=6,
        )


# ---------------------------------------------------------------------------
# TwoRoom
# ---------------------------------------------------------------------------


class TwoRoomAdapter(DatasetAdapter):
    name = "tworoom"

    def build_state_alignment(self) -> StateAlignmentSpec:
        proprio_source = "proprio"

        self.require_columns(proprio_source)
        transform = get_tworoom_state_alignment_transform(
            self.dataset,
            source=proprio_source,
        )
        return StateAlignmentSpec(
            target_transform=transform,
            output_dim=2,
        )


# ---------------------------------------------------------------------------
# PingPong
# ---------------------------------------------------------------------------


class PingPongAdapter(DatasetAdapter):
    """Build the explicitly ordered PingPong state target."""

    name = "pingpong"

    def build_state_alignment(self) -> StateAlignmentSpec:
        effector_position_source = "proprio_effector_pos"
        effector_velocity_source = "proprio_effector_vel"
        gripper_source = "proprio_gripper_opening"
        target_position_source = "privileged_target_box_pos"
        ball_position_source = "privileged_ball_pos"
        ball_velocity_source = "privileged_ball_vel"

        self.require_columns(
            effector_position_source,
            effector_velocity_source,
            gripper_source,
            target_position_source,
            ball_position_source,
            ball_velocity_source,
        )

        transform = get_pingpong_state_alignment_transform(
            self.dataset,
            effector_position_source=effector_position_source,
            effector_velocity_source=effector_velocity_source,
            gripper_source=gripper_source,
            target_position_source=target_position_source,
            ball_position_source=ball_position_source,
            ball_velocity_source=ball_velocity_source,
        )
        return StateAlignmentSpec(
            target_transform=transform,
            output_dim=16,
        )


# ---------------------------------------------------------------------------
# Adapter selection and common pipeline
# ---------------------------------------------------------------------------


ADAPTERS = {
    "cube": CubeAdapter,
    "reacher": ReacherAdapter,
    "pusht": PushTAdapter,
    "tworoom": TwoRoomAdapter,
    "pingpong": PingPongAdapter,
}


def create_dataset_adapter(dataset, data_config) -> DatasetAdapter:
    adapter_config = data_config.adapter

    keys_to_load = list(data_config.dataset.keys_to_load)
    adapter_name = str(adapter_config["type"])

    if adapter_name not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(
            f"unknown dataset adapter {adapter_name!r}; available: {available}"
        )
    return ADAPTERS[adapter_name](dataset, adapter_config, keys_to_load)


def build_data_pipeline(
    dataset,
    data_config,
    img_size: int,
) -> DataPipelineSpec:
    adapter = create_dataset_adapter(dataset, data_config)
    state_alignment = adapter.build_state_alignment()

    transform = spt.data.transforms.Compose(
        get_img_preprocessor(
            source="pixels",
            target="pixels",
            img_size=img_size,
        ),
        state_alignment.target_transform,
        get_column_normalizer(dataset, "action", "action"),
    )

    return DataPipelineSpec(
        transform=transform,
        action_dim=int(dataset.get_dim("action")),
        state_alignment=state_alignment,
    )
