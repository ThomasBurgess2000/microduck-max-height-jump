from types import SimpleNamespace
from xml.etree import ElementTree

import torch
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.distributions import BoundedGaussianDistribution
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_XML,
    MICRODUCK_ALLCOLLISIONS_XML,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    DURABLE_FOOT_DISTANCE_MIN_M,
    DURABLE_HEAD_MAX_ERROR_RAD,
    DURABLE_POSE_RMS_MAX_RAD,
    MicroduckJumpRlCfg,
    make_microduck_jump_env_cfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import make_microduck_roulade_env_cfg


def test_jump_tasks_are_registered_with_backlash_twin():
    tasks = set(list_tasks())
    assert "Mjlab-Jump-Flat-MicroDuck" in tasks
    assert "Mjlab-Jump-Flat-Backlash-MicroDuck" in tasks


def test_reward_stack_makes_visible_height_primary_and_confirms_it_at_horizon():
    cfg = make_microduck_jump_env_cfg()
    assert not any(name.startswith("roulade_") for name in cfg.rewards)
    for name in (
        "jump_launch_velocity",
        "jump_takeoff",
        "jump_visible_height",
        "jump_airtime",
        "jump_touchdown",
        "jump_recovery",
        "jump_settle",
        "jump_stable_landing",
        "jump_durable_landing",
        "jump_durable_visible_height",
    ):
        assert cfg.rewards[name].weight > 0.0
    assert cfg.rewards["jump_durable_visible_height"].weight > cfg.rewards["jump_visible_height"].weight
    assert cfg.rewards["jump_visible_height"].params["cap_at_target"] is False
    assert cfg.rewards["jump_durable_visible_height"].params["log_scale"] is True
    assert cfg.rewards["jump_visible_height"].params["target_height"] == 0.012
    assert cfg.rewards["jump_durable_visible_height"].params["target_height"] == 0.012
    assert cfg.rewards["jump_launch_velocity"].params["target_velocity"] == 0.60


def test_penalty_signs_follow_function_conventions():
    cfg = make_microduck_jump_env_cfg()
    for positive_cost in (
        "nonfoot_impact",
        "airborne_tilt",
        "airborne_drift",
        "dof_pos_limits",
        "self_collisions",
        "failed_landing",
        "post_touchdown_pose",
        "post_touchdown_head_pose",
        "post_touchdown_stance_width",
        "post_touchdown_motion",
    ):
        assert cfg.rewards[positive_cost].weight < 0.0, positive_cost
    # Discovery begins tax-free; curricula add these only after skill emergence.
    assert cfg.rewards["action_rate_huber"].weight == 0.0
    assert cfg.rewards["gentle_landing"].weight == 0.0
    assert cfg.curriculum["action_rate_weight"].func.__name__ == (
        "jump_action_performance_curriculum"
    )
    assert microduck_mdp._jump_v4_action_weight(0.0) == 0.0
    assert microduck_mdp._jump_v4_action_weight(0.30) == -0.02


def test_takeoff_and_body_contact_sensors_are_structural_gates():
    cfg = make_microduck_jump_env_cfg()
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    assert set(sensors) == {
        "feet_ground_contact",
        "nonfoot_ground_contact",
        "self_collision",
    }
    assert sensors["feet_ground_contact"].track_air_time is True
    assert "left_foot_collision" in sensors["feet_ground_contact"].primary.pattern
    assert "right_foot_collision" in sensors["feet_ground_contact"].primary.pattern
    protected = sensors["nonfoot_ground_contact"].primary.pattern
    assert sensors["nonfoot_ground_contact"].primary.mode == "body"
    assert "trunk_base" in protected and "jaw_soft" in protected
    assert "ankle_left" not in protected and "ankle_right" not in protected


def test_both_jump_models_have_fourteen_invisible_sole_envelope_probes():
    for path in (MICRODUCK_ALLCOLLISIONS_XML, MICRODUCK_ALLCOLLISIONS_BACKLASH_XML):
        root = ElementTree.parse(path).getroot()
        probes = [
            site
            for site in root.findall(".//site")
            if site.attrib.get("name", "").startswith("passive_")
            and "sole_probe" in site.attrib.get("name", "")
        ]
        assert len(probes) == 14, path
        assert all(site.attrib.get("rgba") == "0 0 0 0" for site in probes)


def test_reset_is_after_joint_reset_and_has_reverse_curriculum():
    cfg = make_microduck_jump_env_cfg()
    names = list(cfg.events)
    assert names.index("set_jump_state") > names.index("reset_robot_joints")
    reset = cfg.events["set_jump_state"].params
    assert reset["standing_prob"] > 0.0
    assert reset["crouch_prob"] > 0.0
    assert reset["descending_prob"] > 0.0
    assert reset["descending_prob"] == 0.30
    assert reset["standing_z_range"][1] <= 0.116
    assert reset["crouch_z_range"][1] <= 0.073
    assert reset["crouch_overrides"][3] > 1.0
    assert reset["crouch_overrides"][12] < -1.0


def test_play_always_starts_from_standing():
    cfg = make_microduck_jump_env_cfg(play=True)
    reset = cfg.events["set_jump_state"].params
    assert reset["standing_prob"] == 1.0
    assert reset["crouch_prob"] == 0.0
    assert reset["descending_prob"] == 0.0
    assert reset["standing_z_range"] == (0.115, 0.115)
    assert "jump_spawn_mix" not in cfg.curriculum
    assert cfg.viewer.azimuth == 90.0
    assert cfg.viewer.elevation == 0.0
    assert cfg.viewer.lookat[2] < 0.0


def test_one_shot_command_disambiguates_launch_from_settle_without_obs_growth():
    cfg = make_microduck_jump_env_cfg()
    command = cfg.commands["twist"]
    assert command.__class__.__name__ == "JumpOneShotCommandCfg"
    assert 0.5 <= command.launch_window_s <= 1.0
    assert cfg.terminations["jump_failed_recovery"].time_out is False
    assert cfg.rewards["failed_landing"].params["recovery_timeout"] < cfg.episode_length_s
    assert cfg.rewards["jump_stable_landing"].params["stable_time"] == 1.0
    assert "jump_success" not in cfg.terminations
    assert "jump_recovery_success" not in cfg.terminations


def test_durable_landing_requires_forward_head_and_home_width_stance():
    good = microduck_mdp._jump_reset_stance_mask(
        pose_rms=torch.tensor([DURABLE_POSE_RMS_MAX_RAD - 0.01]),
        head_max_error=torch.tensor([DURABLE_HEAD_MAX_ERROR_RAD - 0.01]),
        foot_distance=torch.tensor([DURABLE_FOOT_DISTANCE_MIN_M + 0.001]),
        pose_rms_max=DURABLE_POSE_RMS_MAX_RAD,
        head_max_error_max=DURABLE_HEAD_MAX_ERROR_RAD,
        foot_distance_min=DURABLE_FOOT_DISTANCE_MIN_M,
    )
    twisted_head = microduck_mdp._jump_reset_stance_mask(
        pose_rms=torch.tensor([0.20]),
        head_max_error=torch.tensor([DURABLE_HEAD_MAX_ERROR_RAD + 0.01]),
        foot_distance=torch.tensor([0.084]),
        pose_rms_max=DURABLE_POSE_RMS_MAX_RAD,
        head_max_error_max=DURABLE_HEAD_MAX_ERROR_RAD,
        foot_distance_min=DURABLE_FOOT_DISTANCE_MIN_M,
    )
    overlapped_feet = microduck_mdp._jump_reset_stance_mask(
        pose_rms=torch.tensor([0.20]),
        head_max_error=torch.tensor([0.10]),
        foot_distance=torch.tensor([DURABLE_FOOT_DISTANCE_MIN_M - 0.001]),
        pose_rms_max=DURABLE_POSE_RMS_MAX_RAD,
        head_max_error_max=DURABLE_HEAD_MAX_ERROR_RAD,
        foot_distance_min=DURABLE_FOOT_DISTANCE_MIN_M,
    )
    assert good.tolist() == [True]
    assert twisted_head.tolist() == [False]
    assert overlapped_feet.tolist() == [False]


def test_launch_command_waits_for_post_reset_support():
    command = object.__new__(microduck_mdp.JumpOneShotCommand)
    command._launch_window_s = 0.75
    command.vel_command_b = torch.zeros((1, 3))
    command._env_ref = SimpleNamespace(
        episode_length_buf=torch.tensor([4]),
        step_dt=0.02,
        _jump_support_latched=torch.tensor([False]),
        _jump_support_latched_step=torch.tensor([-1]),
        _jump_descending_spawn=torch.tensor([False]),
        _jump_landed_latched=torch.tensor([False]),
        num_envs=1,
        device="cpu",
    )
    command._env = command._env_ref
    command.compute(0.02)
    assert command.vel_command_b[0, 0].item() == 0.0

    command._env_ref._jump_support_latched[:] = True
    command._env_ref._jump_support_latched_step[:] = 4
    command.compute(0.02)
    assert command.vel_command_b[0, 0].item() == 1.0

    command._env_ref._jump_landed_latched[:] = True
    command.compute(0.02)
    assert command.vel_command_b[0, 0].item() == 0.0


def test_takeoff_gate_rejects_spawn_motion_without_support():
    common = {
        "pre_takeoff": torch.tensor([True]),
        "any_ground": torch.tensor([False]),
        "body_contact": torch.tensor([False]),
        "com_vz": torch.tensor([1.0]),
        "cos_tilt": torch.tensor([1.0]),
        "takeoff_vz_min": 0.05,
        "takeoff_tilt_max_deg": 30.0,
    }
    rejected = microduck_mdp._jump_qualified_takeoff_mask(
        support_latched=torch.tensor([False]), **common
    )
    accepted = microduck_mdp._jump_qualified_takeoff_mask(
        support_latched=torch.tensor([True]), **common
    )
    assert rejected.tolist() == [False]
    assert accepted.tolist() == [True]


def test_visible_takeoff_requires_sustained_clearance_and_airtime():
    common = {
        "airborne": torch.tensor([True]),
        "descending_spawn": torch.tensor([False]),
        "valid_takeoff_latched": torch.tensor([False]),
        "clearance_steps_min": 2,
        "airtime_min": 0.08,
    }
    too_short = microduck_mdp._jump_visible_takeoff_mask(
        clearance_steps=torch.tensor([2]),
        airtime=torch.tensor([0.06]),
        **common,
    )
    too_close = microduck_mdp._jump_visible_takeoff_mask(
        clearance_steps=torch.tensor([1]),
        airtime=torch.tensor([0.10]),
        **common,
    )
    accepted = microduck_mdp._jump_visible_takeoff_mask(
        clearance_steps=torch.tensor([2]),
        airtime=torch.tensor([0.08]),
        **common,
    )
    assert too_short.tolist() == [False]
    assert too_close.tolist() == [False]
    assert accepted.tolist() == [True]


def test_effective_sole_clearance_removes_supported_tiptoe_height():
    clearance, reference = microduck_mdp._jump_effective_sole_clearance(
        sole_height=torch.tensor([0.009]),
        grounded_sole_height=torch.tensor([0.006]),
        pre_takeoff=torch.tensor([True]),
        both_grounded=torch.tensor([True]),
    )
    assert clearance.tolist() == [0.0]
    airborne_clearance, airborne_reference = microduck_mdp._jump_effective_sole_clearance(
        sole_height=torch.tensor([0.013]),
        grounded_sole_height=reference,
        pre_takeoff=torch.tensor([False]),
        both_grounded=torch.tensor([False]),
    )
    assert torch.allclose(airborne_clearance, torch.tensor([0.004]))
    assert torch.equal(airborne_reference, reference)


def test_actor_observation_layout_matches_dynamic_policy_family_exactly():
    jump = make_microduck_jump_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    for group in ("actor", "critic"):
        assert list(jump.observations[group].terms) == list(roulade.observations[group].terms)
    actor = jump.observations["actor"].terms
    assert "base_lin_vel" not in actor
    assert actor["head_command"].params["dim"] == 4
    assert actor["body_command"].params["dim"] == 6


def test_v4_curriculum_is_performance_gated_and_height_has_no_iteration_cliff():
    cfg = make_microduck_jump_env_cfg()
    assert "jump_height_target" not in cfg.curriculum
    assert "jump_stable_height_target" not in cfg.curriculum
    assert cfg.curriculum["jump_spawn_mix"].func.__name__ == (
        "jump_spawn_performance_curriculum"
    )
    assert cfg.curriculum["com_range"].func.__name__ == (
        "jump_dr_performance_curriculum"
    )
    assert microduck_mdp._jump_v4_spawn_mix(0.0) == (0.35, 0.35, 0.30)
    assert microduck_mdp._jump_v4_spawn_mix(0.60) == (0.65, 0.35, 0.0)
    assert microduck_mdp._jump_v4_dr_range(0.0, 0.015) == 0.003
    assert microduck_mdp._jump_v4_dr_range(0.50, 0.015) == 0.015


def test_v5_logs_physical_jump_outcomes_directly():
    cfg = make_microduck_jump_env_cfg()
    for name in (
        "valid_takeoff_rate",
        "max_com_rise_mm",
        "max_bilateral_sole_clearance_mm",
        "max_visible_jump_height_mm",
        "stable_landing_rate",
        "stable_jump_rate",
        "stable_jump_height_mm",
        "durable_jump_rate",
        "durable_visible_jump_height_mm",
        "post_touchdown_survival_s",
        "body_contact_rate",
        "reset_stance_rate",
        "head_max_error_deg",
        "foot_distance_mm",
    ):
        assert name in cfg.metrics
        assert cfg.metrics[name].reduce == "last"


def test_v4_huber_action_rate_is_bounded_and_averaged():
    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            action=torch.full((2, 14), 100.0),
            prev_action=torch.zeros((2, 14)),
        )
    )
    value = microduck_mdp.jump_action_rate_huber(env, delta=0.5, max_error=2.0)
    assert value.shape == (2,)
    assert torch.allclose(value, torch.full((2,), 0.875))


def test_v4_exploration_distribution_is_smoothly_bounded():
    distribution = BoundedGaussianDistribution(
        output_dim=14, init_std=0.6, min_std=0.05, max_std=1.5
    )
    mean = torch.zeros((4, 14))
    distribution.update(mean)
    assert torch.allclose(distribution.std, torch.full_like(mean, 0.6), atol=1e-6)
    distribution.raw_std_param.data.fill_(100.0)
    distribution.update(mean)
    assert torch.all(distribution.std <= 1.5)
    distribution.raw_std_param.data.fill_(-100.0)
    distribution.update(mean)
    assert torch.all(distribution.std >= 0.05)


def test_runner_saves_early_checkpoints_and_is_an_overnight_budget():
    assert MicroduckJumpRlCfg.experiment_name == "microduck_jump"
    assert MicroduckJumpRlCfg.save_interval == 100
    assert MicroduckJumpRlCfg.max_iterations == 5_000
    assert "v6_reset_to_stand" in MicroduckJumpRlCfg.run_name
    assert MicroduckJumpRlCfg.actor.obs_normalization is True
    distribution = MicroduckJumpRlCfg.actor.distribution_cfg
    assert distribution["class_name"].endswith(":BoundedGaussianDistribution")
    assert distribution["init_std"] == 0.6
    assert distribution["max_std"] == 1.5
    assert MicroduckJumpRlCfg.algorithm.entropy_coef == 0.001


def test_bam_startup_and_nonaccumulating_friction_dr_are_present():
    cfg = make_microduck_jump_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert "randomize_joint_friction" in cfg.events
    assert cfg.events["randomize_joint_friction"].func.__name__ == "randomize_bam_friction"
