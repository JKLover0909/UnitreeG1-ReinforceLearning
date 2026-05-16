# Train
## Kiem tra danh sach cac Task
cd /home/jkl/Code/UnitreeG1-ReinforceLearning/unitree_rl_lab
./unitree_rl_lab.sh -l # This is a faster version than isaaclab

## Chay Training theo cac Task
cd /home/jkl/Code/UnitreeG1-ReinforceLearning/unitree_rl_lab
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
 ###same as
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity

## Chay Play theo cac Task Inference with a trained agent:
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity # support for autocomplete task-name
 ###same as
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity

cd /home/jkl/Code/UnitreeG1-ReinforceLearning/unitree_rl_lab
conda activate unitree_rl
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --checkpoint /home/jkl/Code/UnitreeG1-ReinforceLearning/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-11_18-13-06/model_5200.pt --headless

# MUJOCO

## Sim2Sim
Installing the unitree_mujoco.

Set the robot at /simulate/config.yaml to g1
Set domain_id to 0
Set enable_elastic_hand to 1
Set use_joystck to 1.
### start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
#### ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
##### 1. press [L2 + Up] to set the robot to stand up
##### 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
##### 3. Press [R1 + X] to run the policy.
##### 4. Click the mujoco window, and then press 9 to disable the elastic band.
## Sim2Real
You can use this program to control the robot directly, but make sure the on-borad control program has been closed.

./g1_ctrl --network eth0 # eth0 is the network interface name.




# Sau khi restore, build lại g1_ctrl theo thứ tự này:

Vào đúng thư mục project controller
cd ~/Code/UnitreeG1-ReinforceLearning/unitree_rl_lab/deploy/robots/g1_29dof

Xóa build cũ để tránh cache lỗi
rm -rf build

Tạo build mới và compile
mkdir build
cd build
cmake ..
make -j$(nproc)

Chạy controller (nhớ ép đúng DDS lib + interface lo)
LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH ./g1_ctrl -n lo

# Build mujuco
cd ~/Code/UnitreeG1-ReinforceLearning/unitree_mujoco/simulate
rm -rf build
mkdir build && cd build
cmake ..
make -j4

Chay voi bien moi truong sach
LD_LIBRARY_PATH=/usr/local/lib ./unitree_mujoco -r g1 -s scene.xml

#Joystick

Quy đổi tên nút:
L1 = LB
L2 = LT
R1 = RB
R2 = RT
Với project này, code đang dùng kiểu Xbox:
LT là cò trái analog (axis), không phải button
RT là cò phải analog (axis)
LB/RB là bumper trái/phải (button)
Mapping này nằm trong physics_joystick.h.

Nên bấm gì theo prompt của g1_ctrl:
Press [L2 + Up] = giữ LT + nhấn D-pad Up
Press [R1 + X] = giữ RB + nhấn nút X
Với output jstest của bạn:
BtnTL/BtnTR chính là LB/RB
LT/RT sẽ làm đổi giá trị axis (thường là trục 2 và 5), không hiện như button on/off


/home/jkl/miniconda3/envs/unitree_rl/bin/python unitree_rl_lab/scripts/rsl_rl/train_arm_hold.py \
  --task Unitree-G1-29dof-Velocity-ArmHold \
  --num_envs 1 \
  --max_iterations 0 \
  --headless