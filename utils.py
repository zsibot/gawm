import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


def get_pusht_state_alignment_transform(dataset, source="state", target="state_align"):
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data)).float()
    data = data[~torch.isnan(data).any(dim=1)]

    pos_indices = [0, 1, 2, 3]
    angle_index = 4
    pos_data = data[:, pos_indices]
    pos_mean = pos_data.mean(0).clone()
    pos_std = pos_data.std(0).clamp_min(1e-6).clone()

    def align_fn(x):
        x = x.float()
        mean = pos_mean.to(device=x.device, dtype=x.dtype)
        std = pos_std.to(device=x.device, dtype=x.dtype)
        pos = (x[..., pos_indices] - mean) / std
        angle = x[..., angle_index]
        angle_sin_cos = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat((pos, angle_sin_cos), dim=-1)

    transform = dt.transforms.WrapTorchTransform(
        align_fn,
        source=source,
        target=target,
    )
    return transform


def get_tworoom_state_alignment_transform(
    dataset, source="proprio", target="state_align"
):
    """Physical state alignment for the TwoRoom navigation task.

    The raw ``proprio`` is the agent's 2D pixel-space position ``(x, y)``
    (verified to be identical to ``pos_agent`` in the official dataset). There
    is no orientation, no velocity, and no other physical quantity, so the
    alignment target is simply the z-scored 2D position. Output dim = 2.
    """
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data)).float()
    data = data[~torch.isnan(data).any(dim=1)]

    mean = data.mean(0).clone()
    std = data.std(0).clamp_min(1e-6).clone()

    def align_fn(x):
        x = x.float()
        m = mean.to(device=x.device, dtype=x.dtype)
        s = std.to(device=x.device, dtype=x.dtype)
        return (x - m) / s

    transform = dt.transforms.WrapTorchTransform(
        align_fn,
        source=source,
        target=target,
    )
    return transform


def get_reacher_state_alignment_transform(
    dataset, source="observation", target="state_align"
):
    """Physical state alignment for the DMControl Reacher task.

    The raw ``observation`` produced by ``DMControlWrapper._obs_to_array`` is
    the concatenation of ``dm_control.suite.reacher.Reacher.get_observation``:

      [0:2]  position  = qpos     (shoulder, wrist hinge angles, unbounded)
      [2:4]  to_target = finger_to_target xy in world coords
      [4:6]  velocity  = qvel

    Design choices (mirroring pushT / cube alignments):

    - Replace each unbounded hinge angle with ``(sin, cos)``. The reacher
      shoulder and wrist joints in ``dm_control/suite/reacher.xml`` have no
      ``range`` attribute, so qpos can wrap arbitrarily; sin/cos removes the
      ``theta`` / ``theta + 2*pi`` ambiguity and makes MSE well-conditioned.
    - Z-score ``to_target`` (a 2D Cartesian displacement); it has a bounded
      but non-trivial range and benefits from standardisation.
    - Append z-scored ``qvel`` for the pair-state decoder.

    The aligned vector has shape ``(8,)``:
      sin/cos(shoulder) (2) + sin/cos(wrist) (2) + to_target z-scored (2).
      + qvel z-scored (2).
    """
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data)).float()
    data = data[~torch.isnan(data).any(dim=1)]

    raw_dim = data.shape[-1]
    if raw_dim != 6:
        raise ValueError(
            f"Reacher state alignment expects raw observation dim=6 "
            f"(qpos 2 + to_target 2 + qvel 2); got {raw_dim}."
        )

    qpos_idx = [0, 1]
    to_target_idx = [2, 3]
    qvel_idx = [4, 5]

    to_target_data = data[:, to_target_idx]
    tt_mean = to_target_data.mean(0).clone()
    tt_std = to_target_data.std(0).clamp_min(1e-6).clone()
    qvel_data = data[:, qvel_idx]
    qv_mean = qvel_data.mean(0).clone()
    qv_std = qvel_data.std(0).clamp_min(1e-6).clone()

    def align_fn(x):
        x = x.float()
        tt_m = tt_mean.to(device=x.device, dtype=x.dtype)
        tt_s = tt_std.to(device=x.device, dtype=x.dtype)
        qv_m = qv_mean.to(device=x.device, dtype=x.dtype)
        qv_s = qv_std.to(device=x.device, dtype=x.dtype)
        qpos = x[..., qpos_idx]
        qpos_sc = torch.stack(
            (torch.sin(qpos), torch.cos(qpos)), dim=-1
        ).reshape(*qpos.shape[:-1], -1)  # (..., 4)
        to_target = (x[..., to_target_idx] - tt_m) / tt_s  # (..., 2)
        qvel = (x[..., qvel_idx] - qv_m) / qv_s  # (..., 2)
        return torch.cat((qpos_sc, to_target, qvel), dim=-1)

    transform = dt.transforms.WrapTorchTransform(
        align_fn,
        source=source,
        target=target,
    )
    return transform


class PingPongStateAlignmentTransform:
    """Build the fixed PingPong physical-state target."""

    def __init__(self, target, normalization_stats):
        self.target = target
        self.normalization_stats = tuple(normalization_stats)

    def __call__(self, sample):
        normalized_columns = []
        for source, mean, std in self.normalization_stats:
            values = torch.as_tensor(sample[source]).float()
            mean = mean.to(device=values.device, dtype=values.dtype)
            std = std.to(device=values.device, dtype=values.dtype)
            normalized_columns.append((values - mean) / std)

        sample[self.target] = torch.cat(normalized_columns, dim=-1)
        return sample


def get_pingpong_state_alignment_transform(
    dataset,
    *,
    effector_position_source,
    effector_velocity_source,
    gripper_source,
    target_position_source,
    ball_position_source,
    ball_velocity_source,
    target="state_align",
):
    """Create the fixed PingPong physical-state alignment transform."""

    sources = (
        effector_position_source,
        effector_velocity_source,
        gripper_source,
        target_position_source,
        ball_position_source,
        ball_velocity_source,
    )

    normalization_stats = []
    for source in sources:
        values = torch.as_tensor(dataset.get_col_data(source)).float()
        if values.ndim == 1:
            values = values.unsqueeze(-1)

        finite_rows = values[torch.isfinite(values).all(dim=1)]
        if finite_rows.shape[0] < 2:
            raise ValueError(f"not enough finite rows to normalize {source!r}")

        mean = finite_rows.mean(dim=0).clone()
        std = finite_rows.std(dim=0).clamp_min(1e-6).clone()
        normalization_stats.append((source, mean, std))

    return PingPongStateAlignmentTransform(
        target=target,
        normalization_stats=normalization_stats,
    )


def _quat_wxyz_to_rot6d(q):
    """Convert MuJoCo (w, x, y, z) quaternions to the continuous 6D rotation
    representation of Zhou et al. (CVPR 2019).

    Returns the first two columns of the rotation matrix, concatenated. This
    representation is free from the quaternion double-cover ambiguity and
    its L2 distance correlates monotonically with the rotation angle, which
    makes it well-suited to MSE regression targets.

    Args:
        q: tensor of shape ``(..., 4)`` with components in ``(w, x, y, z)``
            order (MuJoCo convention).

    Returns:
        tensor of shape ``(..., 6)``.
    """
    w, x, y, z = q.unbind(dim=-1)
    # Columns of the rotation matrix.
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    return torch.stack((r00, r10, r20, r01, r11, r21), dim=-1)


class CubeStateAlignmentTransform:
    """Build a physical target from OGBench observation and simulator qvel.

    Configuration and velocity groups are concatenated. For ``N`` cubes, the
    output is:

    ``[configuration (12 + 9N), arm qvel (6), cube qvel (6N)]``.

    MuJoCo stores the robot in ``qvel[0:14]`` and each cube as a six-DoF free
    joint starting at ``qvel[14]``.  The gripper linkage/mimic DoFs in
    ``qvel[6:14]`` are deliberately not used as independent state targets.
    """

    def __init__(
        self,
        *,
        observation_source,
        qvel_source,
        target,
        num_cubes,
        znorm_indices,
        znorm_mean,
        znorm_std,
        eff_yaw_indices,
        cube_quat_indices,
        arm_velocity_mean,
        arm_velocity_std,
        cube_velocity_indices,
        cube_velocity_mean,
        cube_velocity_std,
    ):
        self.observation_source = observation_source
        self.qvel_source = qvel_source
        self.target = target
        self.num_cubes = int(num_cubes)
        self.znorm_indices = znorm_indices
        self.znorm_mean = znorm_mean
        self.znorm_std = znorm_std
        self.eff_yaw_indices = eff_yaw_indices
        self.cube_quat_indices = cube_quat_indices
        self.arm_velocity_indices = torch.arange(6, 12, dtype=torch.long)
        self.arm_velocity_mean = arm_velocity_mean
        self.arm_velocity_std = arm_velocity_std
        self.cube_velocity_indices_raw = cube_velocity_indices
        self.cube_velocity_mean = cube_velocity_mean
        self.cube_velocity_std = cube_velocity_std

    @staticmethod
    def _to(tensor, reference, *, dtype=None):
        return tensor.to(
            device=reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )

    def align(self, observation, qvel):
        observation = observation.float()

        z_idx = self._to(self.znorm_indices, observation, dtype=torch.long)

        z_mean = self._to(self.znorm_mean, observation)
        z_std = self._to(self.znorm_std, observation)
        yaw_idx = self._to(self.eff_yaw_indices, observation, dtype=torch.long)

        quat_idx = self._to(self.cube_quat_indices, observation, dtype=torch.long)

        z_part = (observation.index_select(-1, z_idx) - z_mean) / z_std
        yaw_part = observation.index_select(-1, yaw_idx)
        quats = observation.index_select(-1, quat_idx.reshape(-1))
        quats = quats.reshape(*observation.shape[:-1], self.num_cubes, 4)
        rot6d = _quat_wxyz_to_rot6d(quats).reshape(
            *observation.shape[:-1], self.num_cubes * 6
        )
        parts = [z_part, yaw_part, rot6d]

        arm_idx = self._to(
            self.arm_velocity_indices, observation, dtype=torch.long
        )
        arm_mean = self._to(self.arm_velocity_mean, observation)
        arm_std = self._to(self.arm_velocity_std, observation)
        arm_velocity = (
            observation.index_select(-1, arm_idx) - arm_mean
        ) / arm_std
        parts.append(arm_velocity)

        qvel = qvel.float()
        expected_qvel_dim = 14 + 6 * self.num_cubes
        if qvel.shape[-1] != expected_qvel_dim:
            raise ValueError(
                f"cube state alignment expected qvel dim {expected_qvel_dim}, "
                f"got {qvel.shape[-1]}"
            )
        if qvel.shape[:-1] != observation.shape[:-1]:
            raise ValueError(
                "observation and qvel must have identical leading shapes; "
                f"got {observation.shape[:-1]} and {qvel.shape[:-1]}"
            )
        cube_idx = self._to(
            self.cube_velocity_indices_raw, qvel, dtype=torch.long
        )
        cube_mean = self._to(self.cube_velocity_mean, qvel)
        cube_std = self._to(self.cube_velocity_std, qvel)
        cube_velocity = (qvel.index_select(-1, cube_idx) - cube_mean) / cube_std
        parts.append(cube_velocity)

        return torch.cat(parts, dim=-1)

    def __call__(self, sample):
        sample[self.target] = self.align(
            sample[self.observation_source], sample[self.qvel_source]
        )
        return sample


def _finite_zscore_stats(data, indices, name):
    selected = data.index_select(1, indices)
    selected = selected[torch.isfinite(selected).all(dim=1)]
    if selected.shape[0] < 2:
        raise ValueError(f"not enough finite rows to fit {name} statistics")
    return selected.mean(0).clone(), selected.std(0).clamp_min(1e-6).clone()


def get_cube_state_alignment_transform(
    dataset,
    source="observation",
    qvel_source="qvel",
    target="state_align",
    num_cubes=1,
):
    """Create the OGBench cube physical-state alignment transform.

    Raw observation layout is ``19 + 9N``.  The unchanged configuration part
    of the aligned target is ordered as follows:

    - ``0:6``: arm joint position (z-scored)
    - ``6:9``: end-effector position (z-scored)
    - next ``1``: gripper opening (z-scored)
    - next ``3N``: cube positions (z-scored, cube-id order)
    - next ``2``: end-effector yaw as cosine/sine
    - next ``6N``: cube orientations as continuous 6D rotations

    Z-scored arm and cube velocities are appended in that order. Loss masks
    remain explicit experiment configuration.
    """
    observation_array = np.asarray(dataset.get_col_data(source))


    observation_data = torch.from_numpy(observation_array).float()
    joint_pos_idx = list(range(0, 6))
    joint_vel_idx = list(range(6, 12))
    eff_pos_idx = list(range(12, 15))
    eff_yaw_idx = [15, 16]
    gripper_open_idx = [17]
    cube_pos_idx = [19, 20, 21]
    cube_quat_idx = [[22, 23, 24, 25]]

    znorm_idx = torch.tensor(
        joint_pos_idx + eff_pos_idx + gripper_open_idx + cube_pos_idx,
        dtype=torch.long,
    )

    joint_vel_idx_t = torch.tensor(joint_vel_idx, dtype=torch.long)

    znorm_mean, znorm_std = _finite_zscore_stats(
        observation_data, znorm_idx, "cube configuration"
    )

    arm_velocity_mean, arm_velocity_std = _finite_zscore_stats(
        observation_data, joint_vel_idx_t, "arm velocity"
    )
    del observation_data, observation_array

    cube_velocity_indices = torch.arange(
        14, 14 + 6 * num_cubes, dtype=torch.long
    )

    qvel_array = np.asarray(dataset.get_col_data(qvel_source))
    qvel_data = torch.from_numpy(qvel_array).float()

    cube_velocity_mean, cube_velocity_std = _finite_zscore_stats(
        qvel_data, cube_velocity_indices, "cube velocity"
    )
    del qvel_data, qvel_array

    return CubeStateAlignmentTransform(
        observation_source=source,
        qvel_source=qvel_source,
        target=target,
        num_cubes=num_cubes,
        znorm_indices=znorm_idx,
        znorm_mean=znorm_mean,
        znorm_std=znorm_std,
        eff_yaw_indices=torch.tensor(eff_yaw_idx, dtype=torch.long),
        cube_quat_indices=torch.tensor(cube_quat_idx, dtype=torch.long),
        arm_velocity_mean=arm_velocity_mean,
        arm_velocity_std=arm_velocity_std,
        cube_velocity_indices=cube_velocity_indices,
        cube_velocity_mean=cube_velocity_mean,
        cube_velocity_std=cube_velocity_std,
    )


class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
