# G1 29DOF Velocity Task - Định nghĩa Policy & Giải thích Cấu hình

## Phần 1: Policy Quy định "Đúng" vs "Sai" như Thế Nào cho RL Học

### Tổng quan: Vòng lặp RL
```
Quan sát (State) → Mạng Policy → Hành động
                       ↓
                 Mô phỏng (Vật lý)
                       ↓
              Phần thưởng (Tín hiệu dạy)
                       ↓
         Policy Học từ Phần thưởng
```

Policy không tự biết "đúng hay sai". Nó học từ **Hàm Phần thưởng (Reward Function)** - một hàm toán học quy định:
- **Phần thưởng Dương (+)**: Hành động đúng, robot nên làm cái này
- **Phần thưởng Âm (-)**: Hành động sai, robot nên tránh cái này

---

## Phần 2: Các Đoạn Code Quy định "Đúng" vs "Sai"

### Vị trí: `/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py` (Dòng 263-365)

### **Hành động ĐÚNG (Phần thưởng Dương)**

#### 1. **Theo dõi Vận tốc Tuyến tính XY** (Dòng 265-270)
```python
track_lin_vel_xy = RewTerm(
    func=mdp.track_lin_vel_xy_yaw_frame_exp,
    weight=1.0,  # ← Trọng số cao nhất: QUAN TRỌNG NHẤT
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
)
```
**Ý nghĩa**: 
- ✅ Robot di chuyển với vận tốc khớp với lệnh
- Nếu lệnh = [0.5, 0, 0] (tiến 0.5 m/s) và robot làm đúng → **+1.0 phần thưởng**
- Hàm dùng kernel mũ: phần thưởng = exp(-(lỗi²)/std²)
- Đây là **NHIỆM VỤ CHÍNH** (weight=1.0, cao nhất)

#### 2. **Theo dõi Vận tốc Góc Z** (Dòng 271-274)
```python
track_ang_vel_z = RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=0.5,  # ← Nhiệm vụ phụ
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)
```
**Ý nghĩa**: 
- ✅ Robot quay với vận tốc góc khớp với lệnh
- Nếu lệnh = [0, 0, 0.1] (quay 0.1 rad/s) và robot làm → **+0.5 phần thưởng**
- Weight 0.5 = ít quan trọng hơn vận tốc tuyến tính

#### 3. **Phần thưởng Sống sót** (Dòng 276)
```python
alive = RewTerm(func=mdp.is_alive, weight=0.15)
```
**Ý nghĩa**: 
- ✅ Robot không ngã
- Mỗi bước còn đứng → **+0.15 phần thưởng**
- Khuyến khích robot giữ thăng bằng

#### 4. **Kiểu Bước Đi** (Dòng 318-327)
```python
gait = RewTerm(
    func=mdp.feet_gait,
    weight=0.5,
    params={
        "period": 0.8,  # ← Tần suất bước mong muốn
        "offset": [0.0, 0.5],
        "threshold": 0.55,
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    },
)
```
**Ý nghĩa**: 
- ✅ Robot đi bộ với kiểu bước định kỳ (trot/pace)
- Period 0.8s = 1.25 bước/giây (tự nhiên cho G1)
- Chân nên chạm đất xen kẽ → **+0.5 phần thưởng**
- Ngăn cách bò hoặc nhảy

#### 5. **Độ cao Chân** (Dòng 336-346)
```python
feet_clearance = RewTerm(
    func=mdp.foot_clearance_reward,
    weight=1.0,
    params={
        "std": 0.05,
        "tanh_mult": 2.0,
        "target_height": 0.1,  # ← Nâng chân 10cm khỏi mặt đất
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    },
)
```
**Ý nghĩa**: 
- ✅ Chân swing nâng cao khỏi mặt đất (ít nhất 0.1m)
- Ngăn chân vấp ngã và bước đi ổn định → **+1.0 phần thưởng**
- Weight 1.0 = ưu tiên cao cho bước đi tốt

---

### **Hành động SAI (Phần thưởng Âm/Hình Phạt)**

#### 1. **Hình Phạt Độ Cao Body** (Dòng 315)
```python
base_height = RewTerm(
    func=mdp.base_height_l2,
    weight=-10,  # ← HÌNH PHẠT LỚN
    params={"target_height": 0.78}
)
```
**Ý nghĩa**: 
- ❌ Nếu độ cao body ≠ 0.78m → **-10 × (lỗi²) phần thưởng**
- target_height=0.78m là tối ưu cho G1
- Ngăn robot cúi gối hoặc đứng quá cao
- Ví dụ: Robot cúi xuống 0.6m → **hình phạt ≈ -10 × 0.18² = -0.324**

#### 2. **Hình Phạt Định hướng Ngang** (Dòng 314)
```python
flat_orientation_l2 = RewTerm(
    func=mdp.flat_orientation_l2,
    weight=-5.0,
    # (Không tham số: dùng căn chỉnh trọng lực mặc định)
)
```
**Ý nghĩa**: 
- ❌ Nếu robot nghiêng/thiên → **-5.0 × (nghiêng²) phần thưởng**
- Giữ body thẳng (song song mặt đất)
- Ngăn ngã sang bên

#### 3. **Hình Phạt Tiếp xúc Không mong muốn** (Dòng 348-354)
```python
undesired_contacts = RewTerm(
    func=mdp.undesired_contacts,
    weight=-1,
    params={
        "threshold": 1,
        "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"]  # ← Tất cả phần NGOÀI chân
        ),
    },
)
```
**Ý nghĩa**: 
- ❌ Nếu vai/tay/tay cảm chạm đất → **-1 phần thưởng**
- Chỉ chân nên chạm đất
- Ngăn ngã hoặc sụp xuống

#### 4. **Hình Phạt Năng lượng** (Dòng 291)
```python
energy = RewTerm(func=mdp.energy, weight=-2e-5)
```
**Ý nghĩa**: 
- ❌ Mô-men × Vận tốc tiêu thụ → bị phạt **-2e-5**
- Khuyến khích đi bộ hiệu quả (tiêu tốn ít năng lượng)
- Ví dụ: 20Nm ở 10 rad/s = 200W → **-0.004 hình phạt**

#### 5. **Hình Phạt Tốc độ Hành động** (Dòng 290)
```python
action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
```
**Ý nghĩa**: 
- ❌ Thay đổi hành động nhanh → **-0.05 × (Δhành động²) phần thưởng**
- Buộc chuyển động mềm mại, liên tục
- Ngăn hành động co giật

#### 6. **Hình Phạt Gia tốc Khớp** (Dòng 289)
```python
joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
```
**Ý nghĩa**: 
- ❌ Gia tốc cao trong khớp → **-2.5e-7 × (gia tốc²) phần thưởng**
- Khuyến khích chuyển động mềm khớp
- Giảm ứng suất cơ học

#### 7. **Hình Phạt Vận tốc Tuyến tính Base Z** (Dòng 287)
```python
base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
```
**Ý nghĩa**: 
- ❌ Chuyển động dọc (nảy lên/xuống) → **-2.0 × (vận tốc_z²) phần thưởng**
- Giữ body nằm ngang (không nảy)
- Ví dụ: 0.1 m/s dọc → **-2.0 × 0.01 = -0.02 hình phạt**

#### 8. **Hình Phạt Độ lệch Khớp** (Dòng 293-311)
```python
# Tay nên ở gần vị trí mặc định
joint_deviation_arms = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.1,
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*",
        ])
    },
)

# Khớp hông/thân không nên uốn nhiều
joint_deviation_waists = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])}
)

# Hip roll/yaw nên tối thiểu (đi bộ thẳng)
joint_deviation_legs = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])}
)
```
**Ý nghĩa**: 
- ❌ Tay vẫy, hông uốn, hông lăn → hình phạt
- Giữ khớp gần vị trí mặc định (tư thế yên)
- Ngăn chuyển động không cần thiết

#### 9. **Hình Phạt Chân Lượt** (Dòng 328-335)
```python
feet_slide = RewTerm(
    func=mdp.feet_slide,
    weight=-0.2,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    },
)
```
**Ý nghĩa**: 
- ❌ Chân lượt khi chạm đất → **-0.2 hình phạt**
- Khuyến khích bám chặt/đặt chân ổn định
- Ngăn trượt trên bề mặt nhẵn

---

## Phần 3: Giải thích Toàn bộ File

### **Mục đích File**
Định nghĩa **môi trường huấn luyện hoàn chỉnh** để dạy robot G1 đi bộ theo lệnh vận tốc.

### **11 Phần chính**

#### **1. COBBLESTONE_ROAD_CFG (Dòng 23-35)**
```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),                    # Sân 8m × 8m
    num_rows=9, num_cols=21,            # Lưới 9×21 ô địa hình
    horizontal_scale=0.1,               # Độ phân giải 10cm
    vertical_scale=0.005,               # Biến thiên độ cao 5mm
    difficulty_range=(0.0, 1.0),        # 0% đến 100% độ khó
)
```
**Vai trò**: Định nghĩa địa hình huấn luyện
- Bề mặt bằng phẳng với ma sát/độ cứng ngẫu nhiên
- Huấn luyện curriculum: bắt đầu dễ, tăng độ khó khi robot học tốt

---

#### **2. RobotSceneCfg (Dòng 38-97)**
```python
class RobotSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(...)    # Thiết lập địa hình
    robot = UNITREE_G1_29DOF_CFG        # Model G1
    height_scanner = RayCasterCfg(...)  # Cảm biến giống Lidar
    contact_forces = ContactSensorCfg(...)  # Cảm biến tiếp xúc chân
    sky_light = AssetBaseCfg(...)       # Chiếu sáng để render
```
**Vai trò**: Thiết lập môi trường vật lý
- Vật lý địa hình (ma sát, độ nảy)
- Định nghĩa body robot
- Cảm biến cho quan sát state

---

#### **3. EventCfg (Dòng 100-175)**
```python
class EventCfg:
    # Sự kiện khởi động (xảy ra 1 lần khi tạo env)
    physics_material = EventTerm(...)   # Ngẫu nhiên hóa tính chất vật liệu
    add_base_mass = EventTerm(...)      # Ngẫu nhiên hóa khối lượng body
    
    # Sự kiện reset (xảy ra ở đầu episode)
    reset_base = EventTerm(...)         # Vị trí ban đầu ngẫu nhiên
    reset_robot_joints = EventTerm(...) # Vị trí khớp ban đầu ngẫu nhiên
    
    # Sự kiện định kỳ (xảy ra tuần hoàn)
    push_robot = EventTerm(...)         # Nhiễu đẩy ngẫu nhiên mỗi 5s
```
**Vai trò**: Ngẫu nhiên hóa miền
- Huấn luyện robot xử lý biến thiên
- Ví dụ: Ma sát khác, khối lượng khác, đẩy ngẫu nhiên
- Ngăn overfitting vào một môi trường

---

#### **4. CommandsCfg (Dòng 178-193)**
```python
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        ranges=Ranges(
            lin_vel_x=(-0.1, 0.1),    # Tiến/lùi 10cm/s
            lin_vel_y=(-0.1, 0.1),    # Trái/phải 10cm/s
            ang_vel_z=(-0.1, 0.1),    # Quay 0.1 rad/s
        ),
        limit_ranges=Ranges(
            lin_vel_x=(-0.5, 1.0),    # Khả năng tối đa
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.2, 0.2),
        ),
        resampling_time_range=(10.0, 10.0),  # Lệnh mới mỗi 10s
    )
```
**Vai trò**: Định nghĩa lệnh nhiệm vụ
- Huấn luyện: vận tốc nhỏ (-0.1 đến +0.1)
- Giới hạn: khả năng vật lý tối đa
- **Đây là vận tốc ĐỊ CHỈ mà robot phải theo**

---

#### **5. ActionsCfg (Dòng 196-200)**
```python
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],              # 29 khớp
        scale=0.25,                      # ← Tỷ lệ an toàn
        use_default_offset=True,         # Hành động là độ lệch từ tư thế mặc định
    )
```
**Vai trò**: Định nghĩa policy xuất ra cái gì
- Policy xuất ra: vị trí khớp mong muốn (tương đối)
- scale=0.25: giới hạn độ lớn hành động đến 25% an toàn
- Bảo vệ robot khỏi lệnh cực đoan

---

#### **6. ObservationsCfg (Dòng 203-260)**
```python
class ObservationsCfg:
    class PolicyCfg(ObsGroup):  # ← Quan sát mạng Actor
        base_ang_vel = ObsTerm(...)     # Vận tốc quay body
        projected_gravity = ObsTerm(...) # Hướng trọng lực (biết có nghiêng không)
        velocity_commands = ObsTerm(...) # Lệnh vận tốc hiện tại
        joint_pos_rel = ObsTerm(...)    # Vị trí khớp hiện tại
        joint_vel_rel = ObsTerm(...)    # Vận tốc khớp hiện tại
        last_action = ObsTerm(...)      # Hành động lần trước
        
        def __post_init__(self):
            self.history_length = 5     # Dùng 5 frame quá khứ
            self.concatenate_terms = True
    
    class CriticCfg(ObsGroup):  # ← Mạng Critic (đánh giá)
        base_lin_vel = ObsTerm(...)     # Vận tốc tuyến tính
        # (tương tự actor, để học hàm giá trị)
```
**Vai trò**: Định nghĩa không gian state/observation
- **Actor thấy**: thông tin nào policy cần để ra quyết định
- **Critic thấy**: thông tin nào giúp ước tính doanh thu mong đợi
- history_length=5: dùng ngữ cảnh thời gian (5 frame = 0.02s quá khứ)

---

#### **7. RewardsCfg (Dòng 263-365)**
```python
class RewardsCfg:
    # ✅ PHẦN THƯỞNG DƯƠNG (mục tiêu)
    track_lin_vel_xy = RewTerm(..., weight=1.0)      # Nhiệm vụ chính
    track_ang_vel_z = RewTerm(..., weight=0.5)       # Nhiệm vụ phụ
    alive = RewTerm(..., weight=0.15)                # Đứng thẳng
    gait = RewTerm(..., weight=0.5)                  # Bước đi theo kiểu
    feet_clearance = RewTerm(..., weight=1.0)        # Nâng chân
    
    # ❌ PHẦN THƯỞNG ÂM (tránh)
    base_height = RewTerm(..., weight=-10)           # Ở 0.78m
    flat_orientation_l2 = RewTerm(..., weight=-5.0)  # Không nghiêng
    energy = RewTerm(..., weight=-2e-5)              # Tiết kiệm năng lượng
    action_rate = RewTerm(..., weight=-0.05)         # Hành động mềm
    joint_acc = RewTerm(..., weight=-2.5e-7)         # Chuyển động mềm
    undesired_contacts = RewTerm(..., weight=-1)     # Không ngã
    feet_slide = RewTerm(..., weight=-0.2)           # Bám chặt mặt đất
    joint_deviation_* = RewTerm(..., weight=-0.1/-1.0)  # Ở tư thế trung lập
```
**Vai trò**: Định nghĩa hàm phần thưởng (tín hiệu dạy)
- Phần thưởng tổng = tổng tất cả các hạng tử có trọng số
- Policy tối ưu hóa: **max(Σ trọng số_i × phần thưởng_i)**
- Trọng số kiểm soát ưu tiên

---

#### **8. TerminationsCfg (Dòng 368-376)**
```python
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
```
**Vai trò**: Khi nào reset episode
- **Hết thời gian**: 20s mỗi episode (dòng 404: `episode_length_s = 20.0`)
- **Ngã**: độ cao < 0.2m → reset
- **Nghiêng quá**: lỗi định hướng > 0.8 rad → reset

---

#### **9. CurriculumCfg (Dòng 379-385)**
```python
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
```
**Vai trò**: Huấn luyện curriculum (tăng độ khó dần)
- **Cấp địa hình**: bắt đầu bằng phẳng, tăng độ gồ ghề dần
- **Cấp vận tốc**: bắt đầu tốc độ thấp, tăng lệnh dần
- Tự động điều chỉnh khi robot hoạt động tốt

---

#### **10. RobotEnvCfg (Dòng 388-410)**
```python
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()
    
    def __post_init__(self):
        self.decimation = 4              # ← Hành động mỗi 4 bước vật lý
        self.episode_length_s = 20.0     # ← Episode 20 giây
        self.sim.dt = 0.005              # ← Bước vật lý 5ms
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        # Tần số cập nhật cảm biến
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
```
**Vai trò**: Lớp cấu hình chính
- **num_envs=4096**: Huấn luyện 4096 robot song song
- **env_spacing=2.5m**: 2.5m giữa mỗi robot
- **decimation=4**: Cập nhật policy mỗi 20ms (4 × 5ms)
- **episode_length=20s**: Mỗi episode 20 giây
- Đây là những gì train.py tải

---

#### **11. RobotPlayEnvCfg (Dòng 413-420)**
```python
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32         # Ít env hơn để suy luận
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Dùng phạm vi vận tốc tối đa (không phạm vi huấn luyện)
```
**Vai trò**: Cấu hình kiểm tra/triển khai
- num_envs=32: Ít môi trường hơn (để tốc độ)
- Dùng phạm vi vận tốc đầy đủ: kiểm tra tốc độ tối đa

---

## Phần 4: Tóm tắt - Vòng Lặp Hoàn chỉnh

```
┌──────────────────────────────────────────────────────────────┐
│ VÒNG LẶP HUẤN LUYỆN (train.py lặp lại điều này)             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. TẠO LỆnh (CommandsCfg)                                   │
│    → "Tiến 0.05 m/s, quay 0.02 rad/s"                      │
│                                                              │
│ 2. ĐỌC QUAN SÁT (ObservationsCfg)                           │
│    → Vị trí khớp, vận tốc, hướng trọng lực, v.v.           │
│                                                              │
│ 3. POLICY RA QUYẾT ĐỊNH (ActionsCfg)                        │
│    → Neural network xuất 29 vị trí khớp mong muốn           │
│                                                              │
│ 4. MỐ PHỎNG & THỰC THI (RobotSceneCfg, EventCfg)           │
│    → Physics engine di chuyển robot dựa hành động          │
│    → Thêm nhiễu ngẫu nhiên (đẩy, thay đổi ma sát)         │
│                                                              │
│ 5. TÍNH PHẦN THƯỞNG (RewardsCfg)                            │
│    → Theo lệnh: robot có theo lệnh không?                  │
│    → Sống: robot còn đứng không?                            │
│    → Năng lượng: robot tiêu tốn quá nhiều không?           │
│    → Tổng = Σ(trọng số × hạng tử)                         │
│                                                              │
│ 6. KIỂM TRA KẾT THÚC (TerminationsCfg)                      │
│    → Hết thời gian? Robot ngã? Quay lại bước 1             │
│                                                              │
│ 7. CẬP NHẬT POLICY (train.py / rsl_rl)                     │
│    → Dùng PPO để cải thiện policy từ phần thưởng           │
│                                                              │
│ 8. CẬP NHẬT CURRICULUM (CurriculumCfg)                      │
│    → Robot hoạt động tốt, tăng độ khó địa hình            │
│    → Robot hoạt động tốt, tăng lệnh vận tốc                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Phần 5: Siêu tham số Chính Để Điều chỉnh

| Tham số | Hiện tại | Tác dụng |
|---------|----------|---------|
| `track_lin_vel_xy weight` | 1.0 | Mức độ ưu tiên theo lệnh vận tốc |
| `alive weight` | 0.15 | Mức độ phần thưởng không ngã |
| `base_height weight` | -10 | Hình phạt độ cao body sai |
| `energy weight` | -2e-5 | Hình phạt tiêu tốn năng lượng |
| `num_envs` | 4096 | Huấn luyện song song (nhiều hơn = nhanh hơn) |
| `episode_length_s` | 20.0 | Thời lượng episode |
| `ranges.lin_vel_x` | (-0.1, 0.1) | Phạm vi vận tốc huấn luyện |
| `target_height` | 0.78 | Độ cao body mong muốn |
| `decimation` | 4 | Tần suất kiểm soát (thấp hơn = cao hơn) |

## Phần 6: Cách Sửa Cho Hành vi Khác

### Ví dụ 1: Huấn luyện Cho Tốc độ
```python
# Tăng phạm vi lệnh vận tốc
ranges=Ranges(lin_vel_x=(-0.5, 1.0), ...)  # Phạm vi lớn hơn

# Tăng trọng số phần thưởng theo dõi
track_lin_vel_xy = RewTerm(..., weight=2.0)  # Là 1.0

# Giảm hình phạt khác
energy = RewTerm(..., weight=-1e-5)  # Ít hình phạt năng lượng
```

### Ví dụ 2: Huấn luyện Cho Hiệu quả
```python
# Tăng hình phạt năng lượng
energy = RewTerm(..., weight=-1e-3)  # Là -2e-5

# Giảm phần thưởng theo dõi (cho phép bước đi hiệu quả chậm)
track_lin_vel_xy = RewTerm(..., weight=0.5)  # Là 1.0
```

### Ví dụ 3: Huấn luyện Cho Độ Mạnh
```python
# Tăng tần suất nhiễu
push_robot.interval_range_s = (2.0, 2.0)  # Đẩy mỗi 2s thay vì 5s

# Thêm ngẫu nhiên hóa khối lượng
add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)  # Phạm vi lớn hơn
```

---

**Ghi chú Cuối**: Policy KHÔNG được lập trình với kiến thức đi bộ. Nó học hoàn toàn từ hàm phần thưởng. Nếu phần thưởng sai, robot học hành vi sai. Kỹ thuật thiết kế phần thưởng là phần quan trọng nhất của huấn luyện RL!
