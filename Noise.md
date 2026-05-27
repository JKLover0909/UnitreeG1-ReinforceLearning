# Noise Và Randomization Trong Task ArmHold

Tài liệu này mô tả các loại nhiễu/randomization hiện có khi train task `Unitree-G1-29dof-Velocity-ArmHold`.

Hai file ArmHold trực tiếp:

- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_arm_hold_env_cfg.py`
- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/events_arm_hold.py`

Tuy nhiên phần lớn noise không nằm trực tiếp trong hai file này. `velocity_arm_hold_env_cfg.py` kế thừa từ velocity task gốc:

- `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`

Vì vậy ArmHold đang dùng gần như toàn bộ observation noise, event randomization, command randomization, reward, termination và curriculum của task velocity gốc. ArmHold chỉ override action space và thêm event set pose tay.

## 1. Tổng Quan

Trong task này có thể chia "nhiễu" thành các nhóm:

- Observation noise: nhiễu cộng trực tiếp vào input của policy.
- Domain randomization vật lý: random ma sát, khối lượng.
- Reset randomization: random pose/vận tốc lúc reset episode.
- Disturbance trong lúc chạy: push robot định kỳ.
- Command randomization: random velocity command mà robot phải follow.
- Curriculum: tăng dần độ khó command/terrain theo training.
- ArmHold pose event: set pose tay cố định, hiện tại chưa random.

Điểm quan trọng: không phải mọi thứ bên dưới đều là "noise" theo nghĩa sensor noise. Một số là randomization của môi trường hoặc command.

## 2. Observation Noise

Observation noise là nhiễu được cộng vào observation mà actor policy nhìn thấy. Nó nằm trong `ObservationsCfg.PolicyCfg` của `velocity_env_cfg.py`.

Hiện tại các observation có noise:

```python
base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
```

### `base_ang_vel`

```python
noise=Unoise(n_min=-0.2, n_max=0.2)
scale=0.2
```

Đây là vận tốc góc của base/thân robot, thường tương ứng với thông tin IMU gyro.

Ý nghĩa:

- Giả lập sai số/noise của IMU.
- Tránh policy phụ thuộc vào tín hiệu angular velocity quá sạch.
- Giúp policy robust hơn khi chuyển sang MuJoCo hoặc robot thật.

Lưu ý: `scale=0.2` là scale của observation, còn `noise=(-0.2, 0.2)` là nhiễu uniform được cộng vào observation term theo cơ chế của Isaac Lab.

### `projected_gravity`

```python
noise=Unoise(n_min=-0.05, n_max=0.05)
```

Đây là vector hướng trọng lực được chiếu vào body frame của robot. Nó phản ánh orientation của base.

Ý nghĩa:

- Giả lập nhiễu orientation từ IMU.
- Làm policy chịu được sai số khi ước lượng roll/pitch.
- Quan trọng cho cân bằng humanoid, vì projected gravity giúp policy biết robot đang nghiêng thế nào.

Nếu noise này quá lớn, policy có thể khó học cân bằng. Nếu quá nhỏ, policy dễ overfit vào simulation sạch.

### `joint_pos_rel`

```python
noise=Unoise(n_min=-0.01, n_max=0.01)
```

Đây là vị trí joint tương đối so với default joint position.

Ý nghĩa:

- Giả lập sai số encoder vị trí khớp.
- Biên độ `0.01 rad` khoảng 0.57 độ, tương đối nhỏ.
- Phù hợp để tạo robustness nhẹ mà không phá observation.

Với ArmHold, observation này vẫn bao gồm trạng thái joint của robot theo config gốc. Tuy nhiên action của policy chỉ điều khiển lower-body joints.

### `joint_vel_rel`

```python
noise=Unoise(n_min=-1.5, n_max=1.5)
scale=0.05
```

Đây là vận tốc joint.

Ý nghĩa:

- Giả lập nhiễu vận tốc khớp, thường noisy hơn position.
- Biên độ `-1.5 -> 1.5 rad/s` khá lớn.
- `scale=0.05` làm observation velocity được scale nhỏ lại trước khi đưa vào policy.

Joint velocity noise lớn giúp policy bớt phụ thuộc vào vận tốc khớp quá chính xác. Nhưng nếu robot học gait bị rung hoặc khó ổn định, đây là một tham số cần kiểm tra.

### Observation Không Có Noise

Các term sau không có noise trực tiếp:

```python
velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
last_action = ObsTerm(func=mdp.last_action)
```

Nghĩa là command velocity và last action được đưa vào policy sạch, không bị cộng noise.

### History Và Corruption

Trong `PolicyCfg.__post_init__`:

```python
self.history_length = 5
self.enable_corruption = True
self.concatenate_terms = True
```

Ý nghĩa:

- `history_length = 5`: policy nhìn lịch sử 5 frame.
- `enable_corruption = True`: bật noise/corruption cho observation group.
- `concatenate_terms = True`: nối các observation term thành một vector lớn.

Trong log training trước đó, actor observation shape là `(395,)`. Con số này lớn vì observation được stack theo history.

### Critic Observation

Critic có group riêng:

```python
class CriticCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
    last_action = ObsTerm(func=mdp.last_action)
```

Critic không có `noise=...` trong các term này.

Ý nghĩa:

- Actor học trong điều kiện noisy observation.
- Critic được dùng privileged/cleaner observation để estimate value ổn định hơn.
- Critic có thêm `base_lin_vel`, còn actor không có trực tiếp.

## 3. Physics Material Randomization

Trong `EventCfg.physics_material`:

```python
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.0),
        "dynamic_friction_range": (0.3, 1.0),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    },
)
```

Event này chạy ở `startup`, tức là khi environment được tạo.

### Static Friction

```python
static_friction_range = (0.3, 1.0)
```

Static friction là ma sát tĩnh, ảnh hưởng khi chân đứng yên trên mặt đất.

Ý nghĩa:

- Ma sát thấp làm robot dễ trượt chân.
- Ma sát cao làm robot bám đất tốt hơn.
- Random từ `0.3` đến `1.0` giúp policy không chỉ học trên một mặt sàn lý tưởng.

### Dynamic Friction

```python
dynamic_friction_range = (0.3, 1.0)
```

Dynamic friction là ma sát trượt, ảnh hưởng khi chân đã bắt đầu trượt.

Ý nghĩa:

- Tạo nhiều điều kiện tiếp xúc khác nhau.
- Giúp gait bớt nhạy với sai khác giữa Isaac Lab, MuJoCo và robot thật.

### Restitution

```python
restitution_range = (0.0, 0.0)
```

Restitution là độ nảy khi va chạm.

Hiện tại không random vì min=max=0.0. Tức là va chạm không có độ nảy.

### Num Buckets

```python
num_buckets = 64
```

Isaac Lab thường bucket hóa vật liệu để tối ưu simulation. Thay vì mỗi env một material hoàn toàn riêng biệt, các giá trị random được chia vào 64 bucket.

## 4. Base Mass Randomization

Trong `EventCfg.add_base_mass`:

```python
add_base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "mass_distribution_params": (-1.0, 3.0),
        "operation": "add",
    },
)
```

Event này random khối lượng của `torso_link`.

Ý nghĩa:

- Mỗi environment có torso nặng/nhẹ khác nhau.
- Robot phải học policy ổn định dù phân bố khối lượng thân thay đổi.
- Đây là domain randomization quan trọng cho sim2real.

Vì:

```python
operation = "add"
mass_distribution_params = (-1.0, 3.0)
```

Nên khối lượng torso được cộng thêm một lượng random từ `-1 kg` tới `+3 kg`.

Lưu ý: hiện tại chỉ random `torso_link`, không random mass của toàn bộ chân/tay.

## 5. External Force/Torque Khi Reset

Trong `EventCfg.base_external_force_torque`:

```python
base_external_force_torque = EventTerm(
    func=mdp.apply_external_force_torque,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "force_range": (0.0, 0.0),
        "torque_range": (-0.0, 0.0),
    },
)
```

Config này tồn tại, nhưng hiện tại không tạo nhiễu thực sự vì cả force và torque đều bằng 0.

Ý nghĩa hiện tại:

- Không có lực ngoài random lúc reset.
- Không có torque ngoài random lúc reset.
- Term này có thể coi là placeholder để bật sau.

Nếu muốn bật, có thể sửa ví dụ:

```python
"force_range": (-50.0, 50.0),
"torque_range": (-10.0, 10.0),
```

Nhưng cần test cẩn thận vì lực quá lớn có thể làm robot ngã nhiều, learning khó hội tụ.

## 6. Reset Base Randomization

Trong `EventCfg.reset_base`:

```python
reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    },
)
```

Event này chạy mỗi lần reset episode.

### Pose Range

```python
pose_range = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "yaw": (-3.14, 3.14),
}
```

Ý nghĩa:

- Robot spawn ở vị trí x random trong khoảng `-0.5 -> 0.5 m`.
- Robot spawn ở vị trí y random trong khoảng `-0.5 -> 0.5 m`.
- Robot spawn với yaw random gần như toàn vòng `-pi -> pi`.

Điều này giúp policy không phụ thuộc vào world-frame cố định.

### Velocity Range

```python
velocity_range = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
```

Hiện tại root velocity lúc reset không random. Robot bắt đầu episode với base velocity bằng 0.

Nếu muốn train robust với trạng thái ban đầu đang chuyển động, có thể mở random velocity, nhưng nên tăng dần.

## 7. Joint Reset Randomization

Trong `EventCfg.reset_robot_joints`:

```python
reset_robot_joints = EventTerm(
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
        "position_range": (1.0, 1.0),
        "velocity_range": (-1.0, 1.0),
    },
)
```

### Position Range

```python
position_range = (1.0, 1.0)
```

Vì min=max=1.0, joint position gần như không bị random theo scale. Robot reset về default joint position.

Với ArmHold, sau đó còn có:

```python
set_upper_body_on_reset
```

Nên các khớp upper body lại được set về pose tay cố định.

### Velocity Range

```python
velocity_range = (-1.0, 1.0)
```

Joint velocity lúc reset được random trong khoảng `-1.0 -> 1.0 rad/s`.

Ý nghĩa:

- Robot không luôn bắt đầu từ trạng thái động học hoàn toàn đứng yên.
- Policy học ổn định hơn khi joint velocity ban đầu có sai lệch.

## 8. Push Disturbance Trong Lúc Chạy

Trong `EventCfg.push_robot`:

```python
push_robot = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 5.0),
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
)
```

Event này chạy theo interval.

```python
interval_range_s = (5.0, 5.0)
```

Tức là cứ mỗi 5 giây robot bị perturb một lần.

```python
velocity_range = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
}
```

Nó không apply force trực tiếp, mà set/thay đổi velocity của root theo x/y.

Ý nghĩa:

- Giả lập robot bị đẩy hoặc bị nhiễu vận tốc.
- Ép policy học phản xạ giữ thăng bằng.
- Rất quan trọng cho humanoid locomotion vì policy đứng/đi đẹp trong môi trường không nhiễu thường chưa đủ robust.

Hiện tại không có push theo yaw/angular velocity trong config này.

## 9. Command Randomization

Trong `CommandsCfg.base_velocity`:

```python
base_velocity = mdp.UniformLevelVelocityCommandCfg(
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,
    heading_command=False,
    debug_vis=True,
    ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.1, 0.1),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_z=(-0.1, 0.1),
    ),
    limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.3, 0.3),
        ang_vel_z=(-0.2, 0.2),
    ),
)
```

Command randomization không phải sensor noise. Đây là random task command mà policy cần follow.

### Resampling Time

```python
resampling_time_range = (10.0, 10.0)
```

Command được sample lại mỗi 10 giây.

Nếu giảm xuống, ví dụ `(3.0, 6.0)`, robot phải đổi hướng/tốc độ thường xuyên hơn. Điều này khó hơn nhưng có thể giúp policy linh hoạt hơn.

### Standing Envs

```python
rel_standing_envs = 0.02
```

Khoảng 2% environment được command đứng yên.

Ý nghĩa:

- Policy vẫn học khả năng đứng cân bằng.
- Nhưng phần lớn env vẫn học velocity tracking.

### Heading Envs

```python
rel_heading_envs = 1.0
heading_command = False
```

`heading_command=False` nghĩa là task không dùng heading command kiểu target heading. Nó dùng angular velocity command `ang_vel_z`.

### Initial Ranges

```python
ranges:
    lin_vel_x = (-0.1, 0.1)
    lin_vel_y = (-0.1, 0.1)
    ang_vel_z = (-0.1, 0.1)
```

Đây là command range ban đầu. Robot lúc đầu học đi rất chậm/quay rất nhẹ.

Điều này giúp training ổn định hơn, vì humanoid chưa biết đứng/đi mà bắt nó chạy nhanh ngay thường dễ fail.

### Limit Ranges

```python
limit_ranges:
    lin_vel_x = (-0.5, 1.0)
    lin_vel_y = (-0.3, 0.3)
    ang_vel_z = (-0.2, 0.2)
```

Đây là giới hạn command cuối cùng khi curriculum tăng lên.

Ý nghĩa:

- Forward speed tối đa: `1.0 m/s`.
- Backward speed tối đa: `-0.5 m/s`.
- Lateral speed: `-0.3 -> 0.3 m/s`.
- Yaw rate: `-0.2 -> 0.2 rad/s`.

## 10. Command Curriculum

Trong `CurriculumCfg`:

```python
lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
```

Curriculum này tăng dần command range từ `ranges` lên `limit_ranges`.

Ý nghĩa:

- Lúc đầu robot chỉ cần học command nhỏ.
- Khi performance tốt hơn, command range được mở rộng.
- Cuối cùng robot phải follow được command trong `limit_ranges`.

Đây là lý do khi play dùng `RobotPlayEnvCfg`, code set:

```python
self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
```

Vì lúc play ta muốn test policy ở command range cuối, không phải range ban đầu nhỏ.

## 11. Terrain Curriculum Và Terrain Randomization

Terrain được config trong `COBBLESTONE_ROAD_CFG`:

```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)
```

Và trong `RobotSceneCfg`:

```python
terrain_type = "generator"
terrain_generator = COBBLESTONE_ROAD_CFG
max_init_terrain_level = COBBLESTONE_ROAD_CFG.num_rows - 1
```

Curriculum:

```python
terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```

Trong `RobotEnvCfg.__post_init__`:

```python
if getattr(self.curriculum, "terrain_levels", None) is not None:
    if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = True
```

Ý nghĩa:

- Terrain generator có hỗ trợ curriculum.
- Có nhiều row/col terrain để chia cấp độ.
- Nhưng hiện tại `sub_terrains` chỉ có `"flat"`.

Vì vậy, ở config hiện tại, terrain curriculum tồn tại nhưng địa hình thực tế vẫn chủ yếu là flat plane. Nếu muốn terrain noise thật sự, cần thêm các terrain như rough, slope, stairs, discrete obstacles.

## 12. Contact Sensor Không Phải Noise Nhưng Ảnh Hưởng Reward

Scene có contact sensor:

```python
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    history_length=3,
    track_air_time=True,
)
```

Sensor này dùng cho reward:

- `feet_slide`
- `feet_clearance`
- `undesired_contacts`
- `gait`

Hiện tại không thấy noise trực tiếp trên contact sensor trong config này. Nhưng contact force phụ thuộc vào physics randomization, friction, terrain và push.

## 13. ArmHold-Specific Pose Event

Trong `velocity_arm_hold_env_cfg.py`:

```python
set_upper_body = EventTerm(
    func=custom_events.set_upper_body_pose,
    mode="startup",
    params={
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
```

Và:

```python
set_upper_body_on_reset = EventTerm(
    func=custom_events.set_upper_body_pose,
    mode="reset",
    params={
        "joint_pose": {
            "left_shoulder_pitch_joint": 0.6,
            "right_shoulder_pitch_joint": 0.6,
            "left_elbow_joint": 1.0,
            "right_elbow_joint": 1.0,
        },
    },
)
```

Hiện tại đây không phải noise. Đây là pose cố định.

Tác dụng:

- Khi startup, set default upper-body pose.
- Khi reset, enforce lại upper-body pose.
- Dùng `events_arm_hold.py` để ghi vào `asset.data.default_joint_pos`.

Trong `events_arm_hold.py`, hàm chính:

```python
set_upper_body_pose(env, env_ids, asset_cfg, joint_pose)
```

Hàm này:

- Tìm joint theo exact name hoặc regex.
- Ghi giá trị vào `asset.data.default_joint_pos`.
- Nếu joint nằm trong action term thì update `_offset`.

Với ArmHold hiện tại, action chỉ gồm:

```python
joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"]
```

Nên các joint tay không nằm trong action output. Vì vậy tay được giữ thông qua default joint position/event, không phải do policy sinh action tay.

## 14. Action Space Của ArmHold

Trong `velocity_arm_hold_env_cfg.py`:

```python
JointPositionAction = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"],
    scale=0.25,
    use_default_offset=True,
)
```

Đây không phải noise, nhưng rất quan trọng.

Task velocity gốc điều khiển toàn bộ joint:

```python
joint_names=[".*"]
```

ArmHold chỉ điều khiển lower body:

```python
hip, knee, ankle
```

Vì vậy policy output từ 29 action xuống 12 action.

`scale=0.25` nghĩa là target joint position được tính kiểu:

```text
target_joint_pos = default_joint_pos + action * 0.25
```

Do `use_default_offset=True`, default pose rất quan trọng. Nếu default pose tay/chân sai giữa Isaac và MuJoCo, deploy có thể lệch posture.

## 15. Những Thứ Hiện Có Nhưng Chưa Tạo Nhiễu Thật

Các config sau tồn tại nhưng hiện tại chưa random đáng kể:

### External Force/Torque

```python
force_range = (0.0, 0.0)
torque_range = (-0.0, 0.0)
```

Không có lực/torque ngoài lúc reset.

### Reset Base Velocity

```python
"x": (0.0, 0.0)
"y": (0.0, 0.0)
"z": (0.0, 0.0)
"roll": (0.0, 0.0)
"pitch": (0.0, 0.0)
"yaw": (0.0, 0.0)
```

Base velocity lúc reset không random.

### Joint Position Reset

```python
position_range = (1.0, 1.0)
```

Joint position reset không random theo scale.

### Restitution

```python
restitution_range = (0.0, 0.0)
```

Độ nảy không random.

### Arm Pose

```python
left_shoulder_pitch_joint = 0.6
right_shoulder_pitch_joint = 0.6
left_elbow_joint = 1.0
right_elbow_joint = 1.0
```

Pose tay cố định, chưa random.

### Terrain

`terrain_generator` có curriculum nhưng `sub_terrains` chỉ có `flat`, nên chưa có rough/stairs/slope randomization thật sự.

## 16. Nếu Muốn Tăng Nhiễu Theo Hướng Hợp Lý

Một thứ tự tăng nhiễu tương đối an toàn:

1. Tăng nhẹ command range hoặc curriculum speed.
2. Tăng push disturbance từ từ.
3. Random thêm base velocity lúc reset với biên độ nhỏ.
4. Random nhẹ joint position reset.
5. Random thêm mass ở nhiều body hơn, không chỉ torso.
6. Thêm terrain rough nhẹ.
7. Thêm random pose tay ArmHold nếu muốn deploy robust với sai lệch pose.

Không nên bật tất cả cùng lúc, vì nếu policy fail thì khó biết nguyên nhân.

## 17. Gợi Ý Random Pose Tay Cho ArmHold

Hiện tại `events_arm_hold.py` nhận:

```python
joint_pose: dict[str, float]
```

Nên mỗi joint chỉ có một giá trị cố định.

Nếu muốn random pose tay, có thể mở rộng để nhận:

```python
"left_shoulder_pitch_joint": (0.4, 0.8)
"right_shoulder_pitch_joint": (0.4, 0.8)
"left_elbow_joint": (0.8, 1.2)
"right_elbow_joint": (0.8, 1.2)
```

Sau đó trong `events_arm_hold.py`, nếu value là tuple/list thì sample uniform theo từng env.

Ý nghĩa:

- Policy không chỉ quen đúng một upper-body pose.
- Giúp robust hơn nếu MuJoCo/robot thật có offset joint khác Isaac.
- Nhưng random quá rộng có thể làm walking khó học hơn vì center of mass và inertia thay đổi.

## 18. Tóm Tắt Các Loại Nhiễu Hiện Có

| Nhóm | Đang bật? | Tham số chính | Tác dụng |
|---|---:|---|---|
| Observation noise: base angular velocity | Có | `(-0.2, 0.2)` | Giả lập IMU gyro noise |
| Observation noise: projected gravity | Có | `(-0.05, 0.05)` | Giả lập orientation noise |
| Observation noise: joint position | Có | `(-0.01, 0.01)` | Giả lập encoder position noise |
| Observation noise: joint velocity | Có | `(-1.5, 1.5)` | Giả lập encoder velocity noise |
| Friction randomization | Có | `0.3 -> 1.0` | Robust với mặt sàn khác nhau |
| Torso mass randomization | Có | `-1 kg -> +3 kg` | Robust với sai khác mass/inertia |
| Reset x/y/yaw | Có | x/y `+-0.5 m`, yaw `+-pi` | Không phụ thuộc vị trí spawn |
| Reset joint velocity | Có | `-1.0 -> 1.0 rad/s` | Robust với trạng thái động ban đầu |
| Push disturbance | Có | mỗi 5s, x/y `+-0.5` | Học phục hồi sau perturbation |
| Command randomization | Có | velocity command random | Học nhiều vận tốc mục tiêu |
| Command curriculum | Có | ranges -> limit_ranges | Tăng dần độ khó |
| Terrain curriculum | Có về config | hiện chỉ flat | Chưa tạo rough terrain thực sự |
| External force reset | Không hiệu quả | force/torque = 0 | Placeholder |
| Reset base velocity | Không | velocity = 0 | Chưa random |
| Arm pose randomization | Không | fixed values | Tay cố định |

## 19. Kết Luận

Task ArmHold hiện tại đã có một mức randomization khá tốt cho locomotion cơ bản: sensor noise, friction, mass, reset pose, joint velocity, push và command curriculum.

Tuy nhiên các nhiễu liên quan riêng tới ArmHold vẫn còn đơn giản:

- Pose tay đang cố định.
- Upper body không có randomization.
- Terrain chưa thực sự đa dạng.
- External force lúc reset đang tắt.

Nếu mục tiêu là deploy sang MuJoCo/sim2real ổn hơn, điểm nên xem tiếp là random pose upper body, kiểm soát default joint offset giữa Isaac/MuJoCo, và thêm domain randomization nhẹ cho actuator/joint stiffness/damping nếu framework deploy có hỗ trợ.
