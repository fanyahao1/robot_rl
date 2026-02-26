"""Custom VecEnv wrapper that adapts the IsaacLab environment to the rsl-rl 3.x TensorDict API."""

import torch
from tensordict import TensorDict

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class TensorDictVecEnvWrapper(RslRlVecEnvWrapper):
    """Extends :class:`RslRlVecEnvWrapper` to be compatible with rsl-rl >= 3.0.

    rsl-rl 3.x changed the :meth:`get_observations` API: the method must now
    return a :class:`tensordict.TensorDict` mapping observation-group names to
    tensors (e.g. ``{"policy": ..., "critic": ...}``).  The upstream IsaacLab
    wrapper still returns the legacy ``(policy_tensor, info_dict)`` tuple.

    This wrapper overrides :meth:`get_observations` and :meth:`step` / :meth:`reset`
    to comply with the new interface.
    """

    def get_observations(self) -> TensorDict:
        """Return all observation groups as a :class:`TensorDict`.

        Returns:
            A TensorDict whose keys are the observation-group names produced by
            the environment's observation manager (e.g. ``"policy"``,
            ``"critic"``).
        """
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()

        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments and return observations as a TensorDict.

        Returns:
            A tuple of ``(obs_tensordict, info_dict)``.
        """
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and return observations as a TensorDict.

        Args:
            actions: Actions to apply. Shape: ``(num_envs, num_actions)``.

        Returns:
            A tuple of ``(obs_tensordict, rewards, dones, extras)``.
        """
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)

        extras["observations"] = obs_dict

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device), rew, dones, extras
