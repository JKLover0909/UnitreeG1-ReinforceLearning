
# Unitree G1 — Hướng dẫn nhanh

Kho lưu trữ này chứa công cụ huấn luyện và mô phỏng cho robot Unitree G1 (29-DOF), sử dụng Isaac Lab / Isaac Sim kết hợp với pipeline huấn luyện RSL-RL.

## Tổng quan

- Script trợ giúp nhanh: `unitree_rl_lab.sh` bao bọc các thao tác thường dùng (liệt kê, train, play).
- Các script chính: `scripts/rsl_rl/train.py`, `scripts/rsl_rl/play.py` và `scripts/rsl_rl/train_arm_hold.py` (wrapper cho biến thể ArmHold).

## Yêu cầu trước khi chạy

- Hệ điều hành: Linux (đã thử nghiệm trên Ubuntu).
- Sử dụng Conda được khuyến nghị. Tạo và kích hoạt môi trường `unitree_rl` bằng:

```bash
conda env create -f IsaacLab/environment.yml -n unitree_rl
conda activate unitree_rl
```

- Cài Isaac Sim / Isaac Lab theo hướng dẫn NVIDIA (phiên bản tương thích rất quan trọng). Nhiều lỗi runtime phát sinh do `warp`/IsaacSim không tương thích — xem phần Khắc phục sự cố.

## Bắt đầu nhanh

1. Liệt kê các Task có sẵn:

```bash
cd unitree_rl_lab
./unitree_rl_lab.sh -l
```

2. Chạy huấn luyện (ví dụ):

```bash
# dùng wrapper tiện lợi
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless

# hoặc chạy trực tiếp (trong môi trường conda)
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity --num_envs 4
```

3. Chạy play / inference với checkpoint:

```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity

python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity \
  --checkpoint /path/to/model.pt --headless
```

4. Biến thể ArmHold (non-destructive wrapper):

```bash
python unitree_rl_lab/scripts/rsl_rl/train_arm_hold.py \
  --task Unitree-G1-29dof-Velocity-ArmHold \
  --num_envs 12288 \
  --max_iterations 2000 \
  --headless
```

Wrapper này đăng ký env `ArmHold` ở runtime rồi chuyển tiếp các tham số sang `train.py`.

## MuJoCo / Sim2Sim

Xây dựng và chạy mô phỏng dựa trên MuJoCo (unitree_mujoco):

```bash
cd unitree_mujoco/simulate
rm -rf build && mkdir build && cd build
cmake ..
make -j4
LD_LIBRARY_PATH=/usr/local/lib ./unitree_mujoco -r g1 -s scene.xml
```

Chạy chương trình điều khiển gốc (`g1_ctrl`):

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./g1_ctrl -n lo
```

## Ánh xạ Joystick (kiểu Xbox)

- L1 = LB (button)
- L2 = LT (analog axis)
- R1 = RB (button)
- R2 = RT (analog axis)

Ghi chú: LT/RT là trigger analog (axis), không phải button on/off. Bản đồ nút nằm trong file `physics_joystick.h`.

## Các lệnh thường dùng / Ví dụ

- Ví dụ resume huấn luyện ArmHold:

```bash
python scripts/rsl_rl/train_arm_hold.py \
  --task Unitree-G1-29dof-Velocity-ArmHold \
  --num_envs 16384 \
  --max_iterations 15000 \
  --resume \
  --load_run 2026-05-16_13-57-42 \
  --checkpoint model_1999.pt \
  --headless
```

- Ví dụ chạy play với checkpoint:

```bash
python scripts/rsl_rl/play_arm_hold.py \
  --task Unitree-G1-29dof-Velocity-ArmHold \
  --num_envs 16 \
  --load_run 2026-05-16_16-24-21 \
  --checkpoint model_6000.pt
```

## Khắc phục sự cố (Troubleshooting)

- Lỗi `IsaacSim / Warp mismatch`: nếu thấy thông báo như sau khi khởi App/extension:

```
module 'warp.types' has no attribute 'array'
module 'warp' has no attribute 'context'
```

thì rất có khả năng bản `warp` hoặc bản IsaacSim đang dùng không tương thích. Một cách kiểm tra nhanh:

```bash
conda activate unitree_rl
python - <<'PY'
import importlib
import sys
print('python:', sys.executable)
try:
    import warp
    print('warp:', warp.__file__)
    print('warp.types sample:', dir(importlib.import_module('warp.types'))[:30])
except Exception as e:
    print('warp error:', e)
PY
```

Hướng khắc phục:

- An toàn nhất: tạo lại môi trường theo hướng dẫn IsaacLab/IsaacSim chính thức (dùng script/requirements của IsaacLab tương ứng với release bạn dùng). Đây là cách đảm bảo `warp` và các extension tương thích.
- Thử thủ công: gỡ cài `warp` hiện tại rồi cài wheel `warp` tương thích với phiên bản IsaacSim (tham khảo release notes của IsaacSim). Cách này rủi ro nếu không đúng bản.

Nếu bạn không chắc, gửi log lỗi đầy đủ cho tôi và tôi sẽ chỉ dẫn lệnh cài phù hợp.

## Cách repo đăng ký môi trường (environments)

- Các module task nằm trong `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/` và được import tự động bằng tiện ích dò package. Để thêm env mới, tạo `*_env_cfg.py` và `*_register.py` (hoặc dùng đăng ký động). File `train_arm_hold.py` là ví dụ cách đăng ký không can thiệp mã gốc.

## Đóng góp

- Thêm task mới dưới `unitree_rl_lab/tasks/` theo mẫu hiện có.
- Dùng `list_envs.py` để kiểm tra env đã được đăng ký:

```bash
python scripts/list_envs.py
```

## Nơi tìm trợ giúp

- Mở issue trong repo này kèm log và các lệnh bạn đã chạy.
- Với lỗi liên quan đến build IsaacSim/warp, tham khảo tài liệu NVIDIA IsaacSim và hướng dẫn cài `warp` tương ứng.

---

## Bảng giải thích tham số (Tham số CLI chính)

| Tham số | Vai trò | Giá trị mặc định |
|---|---|---:|
| `--task` | Tên task (môi trường) để chạy, ví dụ `Unitree-G1-29dof-Velocity` | `None` (phải chỉ định hoặc dùng mặc định trong script) |
| `--num_envs` | Số lượng environment chạy song song (vectorized environments) | `None` (dùng `env_cfg.scene.num_envs` nếu không chỉ định) |
| `--max_iterations` | Số vòng học tối đa (số iteration của runner) | `None` (dùng `agent_cfg.max_iterations` nếu không chỉ định) |
| `--seed` | Giá trị seed ngẫu nhiên cho môi trường/agent | `None` |
| `--distributed` | Bật chế độ huấn luyện phân tán (multi-GPU / multi-node) | `False` |
| `--video` | Bật ghi video khi huấn luyện | `False` |
| `--video_length` | Độ dài video (số bước) | `200` |
| `--video_interval` | Khoảng (số bước) giữa các lần ghi video | `2000` |
| `--experiment_name` | Tên thư mục experiment lưu logs | `None` (nếu rỗng sẽ lấy từ tên task) |
| `--run_name` | Tên phụ (suffix) cho run hiện tại | `None` |
| `--resume` | Bật resume từ checkpoint | `False` |
| `--load_run` | Tên thư mục run để tải khi resume | `None` |
| `--checkpoint` | File checkpoint để tải | `None` |
| `--logger` | Chọn module logger (`wandb`, `tensorboard`, `neptune`) | `None` |
| `--log_project_name` | Tên project cho `wandb` / `neptune` | `None` |
| `--headless` | Chạy IsaacSim không có GUI (kiểm soát hiển thị) | `False` |
| `--livestream` | Bật livestream: `0` tắt, `1` public, `2` private, `-1` dùng ENV | `-1` |
| `--enable_cameras` | Bật camera sensor ngay cả khi headless | `False` |
| `--xr` | Bật chế độ XR (VR/AR) | `False` |
| `--device` | Thiết bị để chạy mô phỏng (`cpu`, `cuda`, `cuda:N`) | `cuda:0` |
| `--experience` | Tên file experience để load cho SimulationApp | `""` (trống, tự chọn dựa trên headless/enable_cameras) |
| `--rendering_mode` | Chế độ rendering (`performance`, `balanced`, `quality`) | `balanced` |
| `--kit_args` | Tham số chuỗi truyền cho Omniverse Kit (chứa nhiều tùy chọn) | `""` |
| `--anim_recording_enabled` | Bật ghi animation time-sampled | `False` |
| `--anim_recording_start_time` | Thời điểm bắt đầu ghi animation (giây) | `0` |
| `--anim_recording_stop_time` | Thời điểm dừng ghi animation (giây) | `10` |

Ghi chú: một số giá trị mặc định (như `env_cfg.scene.num_envs` hay `agent_cfg.max_iterations`) được thiết lập trong file cấu hình nhiệm vụ (`*_env_cfg.py` và cấu hình runner). Nếu bạn thay đổi tham số trong CLI, giá trị CLI sẽ ghi đè cấu hình.

---

Nếu bạn muốn, tôi có thể tiếp tục và:

- Thêm checklist chi tiết để tái tạo môi trường (conda) tự động,
- Tạo thư mục `docs/` với hướng dẫn từng bước và link tham khảo,
- Hoặc tạo script tự động dựng môi trường conda tương thích với IsaacSim mà bạn đang dùng.

