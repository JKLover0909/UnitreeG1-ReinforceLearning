from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from unitree_rl_lab.tasks.locomotion import mdp
from . import velocity_env_cfg as base
from unitree_rl_lab.tasks.locomotion.mdp import events_arm_hold as custom_events

@configclass
class RobotEnvCfg(base.RobotEnvCfg):
    """
    Inherit velocity env but:
    - Restrict JointPositionAction to lower-body joints only (hip/knee/ankle)
    - Add startup/reset event to set upper-body joint poses (arms forward)
    """

    @configclass
    class ActionsCfg(base.ActionsCfg):
        # allow only lower-body joints for policy control (regex patterns)
        JointPositionAction = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"],
            scale=0.25,
            use_default_offset=True,
        )

    @configclass
    class EventCfg(base.EventCfg):
        # inherit base events by subclassing; add our set_upper_body pose event
        set_upper_body = EventTerm(
            func=custom_events.set_upper_body_pose,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                # patterns or exact names; example angles in radians
                "joint_pose": {
                    "left_shoulder_pitch_joint": 0.6,
                    "right_shoulder_pitch_joint": 0.6,
                    "left_elbow_joint": 1.0,
                    "right_elbow_joint": 1.0,
                    "left_wrist_roll_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                },
            },
        )
        # optionally also enforce on reset
        set_upper_body_on_reset = EventTerm(
            func=custom_events.set_upper_body_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_pose": {
                    "left_shoulder_pitch_joint": 0.6,
                    "right_shoulder_pitch_joint": 0.6,
                    "left_elbow_joint": 1.0,
                    "right_elbow_joint": 1.0,
                },
            },
        )

    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()


@configclass
class RobotPlayEnvCfg(RobotEnvCfg, base.RobotPlayEnvCfg):
    # keep play overrides (if needed)
    pass
