"""Microduck one-shot standing jump task.

A supported upward launch must produce real contact loss, airborne
height/airtime only pay new max-so-far progress, and the largest reward is
reserved for a quiet, centered two-foot landing.  The existing twist-vx command
slot carries a binary one-shot request (1 = launch, 0 = settle), removing the
otherwise ambiguous standing observation before launch versus after landing.
There is no pose trajectory or reward waypoint for the policy to camp on.
"""

import math
from copy import deepcopy

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)


EPISODE_LENGTH_S = 3.5
STAND_Z = 0.115
LAUNCH_WINDOW_S = 0.75
RECOVERY_TIMEOUT_S = 1.40
DURABLE_SETTLE_TIME_S = 1.00
DURABLE_POSE_RMS_MAX_RAD = microduck_mdp.JUMP_DURABLE_POSE_RMS_MAX_RAD
DURABLE_HEAD_MAX_ERROR_RAD = microduck_mdp.JUMP_DURABLE_HEAD_MAX_ERROR_RAD
DURABLE_FOOT_DISTANCE_MIN_M = microduck_mdp.JUMP_DURABLE_FOOT_DISTANCE_MIN_M
TAKEOFF_CLEARANCE_MIN_M = 0.003
TAKEOFF_AIRTIME_MIN_S = 0.08
# This is a normalization scale, not a pay cap. The CEM physics sweep writes
# its measured ceiling separately; V5 remains rewarded above this value.
# Nominal 7.4 V BAM CEM sweep: 13.67 mm reproducible ceiling.  Use 12 mm as
# the normalization scale (not a cap); the frontier reward stays uncapped so
# every additional millimetre remains valuable.
VISIBLE_HEIGHT_SCALE_M = 0.012
LAUNCH_VELOCITY_SCALE_M_S = 0.60
STANDING_Z_RANGE = (0.112, 0.115)
CROUCH_Z_RANGE = (0.068, 0.071)
FEET_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

# Foot-flat crouch measured by the CPU MuJoCo feasibility sweep. The signs are
# the canonical 14-servo layout; reset_jump_state resolves the actual entity
# joint ids so the backlash model remains correct.
CROUCH_OVERRIDES = {
    2: 0.4188,
    3: 1.3776,
    4: 0.9588,
    11: -0.4188,
    12: -1.3776,
    13: -0.9588,
}


def make_microduck_jump_env_cfg(play: bool = False):
    """Build the all-collision jump environment from the dynamic-task recipe."""
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Contact on any body except the two ankle/sole bodies is a failed/violent
    # landing. Most collision geoms in the exported MJCF are intentionally
    # unnamed, so this must select named BODIES rather than use a geom regex.
    nonfoot_ground = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=(
                r"^(trunk_base|yaw2roll|hip_l|upper_leg_left|leg|neck|neck_pitch|"
                r"yaw_roll_motion|jaw_soft|bearing_roll|hip_l_2|upper_leg_right|leg_2)$"
            ),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (feet_ground, nonfoot_ground, self_collision)

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = 1.0

    # One-shot task flag in the existing twist-vx observation slot.  The v1
    # policy had no way to distinguish the same upright pose at episode start
    # (jump now) from after touchdown (stay still), which produced repeated
    # launch/recovery thrashing.  This preserves the 61D hot-swap contract.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.JumpOneShotCommandCfg(
        **{**vars(command), "launch_window_s": LAUNCH_WINDOW_S}
    )

    # Fresh reward dictionary makes it impossible for a roll-completion annuity
    # or walking term to leak into this task.
    cfg.rewards = {
        "jump_launch_velocity": RewardTermCfg(
            func=microduck_mdp.jump_launch_velocity_progress,
            weight=2.0,
            params={
                "target_velocity": LAUNCH_VELOCITY_SCALE_M_S,
                "feet_cfg": deepcopy(FEET_CFG),
            },
        ),
        "jump_takeoff": RewardTermCfg(
            func=microduck_mdp.jump_takeoff_bonus,
            weight=1.0,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "jump_visible_height": RewardTermCfg(
            func=microduck_mdp.jump_visible_height_progress,
            weight=6.0,
            params={
                "target_height": VISIBLE_HEIGHT_SCALE_M,
                "cap_at_target": False,
                "feet_cfg": deepcopy(FEET_CFG),
            },
        ),
        "jump_airtime": RewardTermCfg(
            func=microduck_mdp.jump_airtime_progress,
            weight=1.0,
            params={"target_airtime": 0.10, "feet_cfg": deepcopy(FEET_CFG)},
        ),
        "jump_touchdown": RewardTermCfg(
            func=microduck_mdp.jump_touchdown_quality,
            weight=3.0,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "jump_recovery": RewardTermCfg(
            func=microduck_mdp.jump_recovery_progress,
            weight=2.0,
            params={"stand_height": STAND_Z, "feet_cfg": deepcopy(FEET_CFG)},
        ),
        # Paid-once max frontier across upright, height, centering, HOME pose,
        # low body/joint speed, and body-clear support.  This supplies a dense
        # bridge to the stable bonus without creating a per-step annuity.
        "jump_settle": RewardTermCfg(
            func=microduck_mdp.jump_settle_progress,
            weight=5.0,
            params={"stand_height": STAND_Z, "feet_cfg": deepcopy(FEET_CFG)},
        ),
        # A modest flat landing bonus keeps discovery reachable. The larger
        # height-confirmation term below makes the highest clean jump the argmax.
        "jump_stable_landing": RewardTermCfg(
            func=microduck_mdp.jump_stable_landing_bonus,
            weight=6.0,
            params={
                "stable_time": DURABLE_SETTLE_TIME_S,
                "feet_cfg": deepcopy(FEET_CFG),
            },
        ),
        "jump_durable_landing": RewardTermCfg(
            func=microduck_mdp.jump_durable_landing_bonus,
            weight=10.0,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "jump_durable_visible_height": RewardTermCfg(
            func=microduck_mdp.jump_durable_visible_height_bonus,
            weight=24.0,
            params={
                "target_height": VISIBLE_HEIGHT_SCALE_M,
                "log_scale": True,
                "feet_cfg": deepcopy(FEET_CFG),
            },
        ),
        # A failed recovery has explicit negative episode mass before its
        # terminal transition, so early crashing cannot avoid accumulated
        # settling costs by shortening the episode.
        "failed_landing": RewardTermCfg(
            func=microduck_mdp.jump_failed_recovery_cost,
            weight=-8.0,
            params={
                "recovery_timeout": RECOVERY_TIMEOUT_S,
                "feet_cfg": deepcopy(FEET_CFG),
            },
        ),
        "post_touchdown_pose": RewardTermCfg(
            func=microduck_mdp.jump_post_touchdown_pose_cost,
            weight=-0.75,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "post_touchdown_head_pose": RewardTermCfg(
            func=microduck_mdp.jump_post_touchdown_head_pose_cost,
            weight=-1.0,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "post_touchdown_stance_width": RewardTermCfg(
            func=microduck_mdp.jump_post_touchdown_stance_width_cost,
            weight=-1.0,
            params={
                "min_distance": DURABLE_FOOT_DISTANCE_MIN_M,
                "feet_cfg": deepcopy(FEET_CFG),
            },
        ),
        "post_touchdown_motion": RewardTermCfg(
            func=microduck_mdp.jump_post_touchdown_motion_cost,
            weight=-0.4,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "nonfoot_impact": RewardTermCfg(
            func=microduck_mdp.body_impact_cost,
            weight=-0.12,
            params={"sensor_name": nonfoot_ground.name, "threshold": 1.0},
        ),
        "airborne_tilt": RewardTermCfg(
            func=microduck_mdp.jump_airborne_tilt_cost,
            weight=-0.4,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "airborne_drift": RewardTermCfg(
            func=microduck_mdp.jump_horizontal_drift_cost,
            weight=-0.5,
            params={"feet_cfg": deepcopy(FEET_CFG)},
        ),
        "dof_pos_limits": deepcopy(cfg.rewards["dof_pos_limits"]),
        # Discovery-time smoothness is deliberately light; stronger polish is
        # introduced only after takeoff/landing have had time to emerge.
        "action_rate_huber": RewardTermCfg(
            func=microduck_mdp.jump_action_rate_huber,
            weight=0.0,
            params={"delta": 0.5, "max_error": 2.0},
        ),
        "joint_torque_rate_l2": RewardTermCfg(
            func=microduck_mdp.joint_torque_rate_l2, weight=0.0
        ),
        "gentle_landing": RewardTermCfg(
            func=microduck_mdp.trunk_vertical_accel_penalty,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.1,
            params={"sensor_name": self_collision.name},
        ),
    }

    # Do not terminate on an apparently stable pose. V4 stopped after 200 ms
    # and hid a delayed fall. V5 runs successful attempts to the horizon and
    # pays the height jackpot only there.
    cfg.terminations.pop("jump_success", None)
    cfg.terminations.pop("jump_recovery_success", None)
    cfg.terminations["jump_failed_recovery"] = TerminationTermCfg(
        func=microduck_mdp.jump_failed_recovery,
        time_out=False,
        params={
            "recovery_timeout": RECOVERY_TIMEOUT_S,
            "feet_cfg": deepcopy(FEET_CFG),
        },
    )

    cfg.events.pop("set_roulade_state", None)
    cfg.events["set_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
        params={
            "standing_prob": 0.35,
            "crouch_prob": 0.35,
            "descending_prob": 0.30,
            "standing_z_range": STANDING_Z_RANGE,
            "crouch_z_range": CROUCH_Z_RANGE,
            "descending_z_range": (0.125, 0.150),
            "descending_vz_range": (-0.35, -0.05),
            "tilt_max_deg": 2.0,
            "crouch_overrides": CROUCH_OVERRIDES,
            "joint_noise_std": 0.03,
        },
    )

    # Every hardening transition is performance-gated. V3's fixed iteration
    # stages widened DR and then switched on raw L2 smoothing before the skill
    # existed, producing an avoidable reward cliff.
    cfg.curriculum.clear()

    cfg.curriculum["jump_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.jump_spawn_performance_curriculum,
        params={
            "event_name": "set_jump_state",
        },
    )
    cfg.curriculum["com_range"] = CurriculumTermCfg(
        func=microduck_mdp.jump_dr_performance_curriculum,
        params={
            "event_name": "randomize_com",
            "max_range": 0.015,
        },
    )
    cfg.curriculum["head_com_range"] = CurriculumTermCfg(
        func=microduck_mdp.jump_dr_performance_curriculum,
        params={
            "event_name": "randomize_head_com",
            "max_range": 0.010,
        },
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.jump_action_performance_curriculum,
        params={
            "reward_name": "action_rate_huber",
        },
    )

    # Physical episode outcomes are logged directly; weighted reward mass is
    # no longer the only way to infer whether a real jump happened.
    cfg.metrics.update(
        {
            "valid_takeoff_rate": MetricsTermCfg(
                func=microduck_mdp.jump_valid_takeoff_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "max_com_rise_mm": MetricsTermCfg(
                func=microduck_mdp.jump_max_com_rise_mm_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "max_bilateral_sole_clearance_mm": MetricsTermCfg(
                func=microduck_mdp.jump_max_bilateral_clearance_mm_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "max_visible_jump_height_mm": MetricsTermCfg(
                func=microduck_mdp.jump_max_visible_height_mm_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "stable_landing_rate": MetricsTermCfg(
                func=microduck_mdp.jump_stable_landing_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "stable_jump_rate": MetricsTermCfg(
                func=microduck_mdp.jump_stable_jump_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "stable_jump_height_mm": MetricsTermCfg(
                func=microduck_mdp.jump_stable_height_mm_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "durable_jump_rate": MetricsTermCfg(
                func=microduck_mdp.jump_durable_jump_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "durable_visible_jump_height_mm": MetricsTermCfg(
                func=microduck_mdp.jump_durable_visible_height_mm_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "post_touchdown_survival_s": MetricsTermCfg(
                func=microduck_mdp.jump_post_touchdown_survival_s_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "body_contact_rate": MetricsTermCfg(
                func=microduck_mdp.jump_body_contact_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "reset_stance_rate": MetricsTermCfg(
                func=microduck_mdp.jump_reset_stance_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "head_max_error_deg": MetricsTermCfg(
                func=microduck_mdp.jump_head_max_error_deg_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
            "foot_distance_mm": MetricsTermCfg(
                func=microduck_mdp.jump_foot_distance_mm_metric,
                params={"feet_cfg": deepcopy(FEET_CFG)},
                reduce="last",
            ),
        }
    )

    if play:
        reset_cfg = cfg.events["set_jump_state"]
        reset_cfg.params.update(
            {
                "standing_prob": 1.0,
                "crouch_prob": 0.0,
                "descending_prob": 0.0,
                "standing_z_range": (STAND_Z, STAND_Z),
                "tilt_max_deg": 0.0,
                "joint_noise_std": 0.0,
            }
        )
        # The training spawn curriculum mutates the live reset term before each
        # reset; leaving it enabled silently overrides the standing-only play
        # config. Evaluation must therefore remove it entirely.
        cfg.curriculum.pop("jump_spawn_mix", None)
        # Exact side profile at near-ground level. This makes contact loss and
        # the landing readable in checkpoint videos without debug overlays.
        cfg.viewer.azimuth = 90.0
        cfg.viewer.elevation = 0.0
        cfg.viewer.distance = 0.55
        cfg.viewer.lookat = (0.0, 0.0, -0.09)
        cfg.viewer.fovy = 35.0
        cfg.viewer.width = 1280
        cfg.viewer.height = 720
    return cfg


MicroduckJumpRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckJumpRlCfg.experiment_name = "microduck_jump"
MicroduckJumpRlCfg.run_name = "microduck_jump_v6_reset_to_stand"
MicroduckJumpRlCfg.save_interval = 100
MicroduckJumpRlCfg.max_iterations = 5_000
MicroduckJumpRlCfg.actor.distribution_cfg = {
    "class_name": "mjlab_microduck.distributions:BoundedGaussianDistribution",
    "init_std": 0.6,
    "min_std": 0.05,
    "max_std": 1.5,
}
MicroduckJumpRlCfg.algorithm.entropy_coef = 0.001
