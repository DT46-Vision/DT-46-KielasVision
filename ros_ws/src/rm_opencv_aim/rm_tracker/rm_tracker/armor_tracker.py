import math
import numpy as np
from rm_tracker.Kalman import MultiDimKalmanFilter

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
EPS = 1e-9


class Armor:
    def __init__(self, armor_id: int, x: float, y: float, z: float):
        self.armor_id = armor_id
        self.x = x  # 相机系，单位：mm（若你是 m，请自行统一）
        self.y = y
        self.z = z


class ColorPrint:
    def __init__(self):
        self.PINK = "\033[38;5;218m"
        self.CYAN = "\033[96m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"


class Tracker:
    """
    输入：ArmorsCppMsg（每个 armor: armor_id, dx, dy, dz）
    过程：选目标 -> IMM(KF) 融合(位姿+速度) -> 相机->枪管 -> 弹道解算 -> 射击判定
    输出：(shoot_flag, yaw_sent, pitch_sent, msg_color)
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.color = ColorPrint()

        # ========== 外参：相机->枪管（单位：m / deg）==========
        # 平移：相机坐标在枪管坐标中的位置（把相机点变到枪管时用：P_gun = R * P_cam + T）
        self.camera_tx_m = 0.0
        self.camera_ty_m = 0.0
        self.camera_tz_m = -0.03
        # 旋转（相机系到枪管系的欧拉角，右手，Z*Y*X）
        self.camera_roll_deg = 0.0
        self.camera_pitch_deg = 0.0
        self.camera_yaw_deg = 4.0

        # ========== 追踪配置 ==========
        self.tracking_color = 0      # 0=红, 1=蓝, 10=不追
        self.follow_decision = 0
        self.track_deep_tol = 10.0   # z 差值阈值（mm）
        self.shoot_yaw_max = 1.5
        self.shoot_pitch_max = 1.0

        # ========== IMM / KF 参数 ==========
        self.kf_sigma_acc = 2.0
        self.kf_sigma_jerk = 20.0
        self.kf_meas_std = 0.01
        self.kf_mu_cv = 0.5
        self.kf_mu_ca = 0.5
        self.kf_trans_00 = 0.95
        self.kf_trans_01 = 0.05
        self.kf_trans_10 = 0.05
        self.kf_trans_11 = 0.95
        self.kf_a_init_std = 3.0  # 新增：初始加速度标准差

        # KF
        self.use_kf = True
        self.frame_add = 60
        self.predict = False
        self.lost = 0
        self.if_find = False

        self.kf = MultiDimKalmanFilter(
            dim=3,
            sigma_acc=self.kf_sigma_acc,
            sigma_jerk=self.kf_sigma_jerk,
            meas_std=self.kf_meas_std,
            mu_init=(self.kf_mu_cv, self.kf_mu_ca),
            P_trans=[
                [self.kf_trans_00, self.kf_trans_01],
                [self.kf_trans_10, self.kf_trans_11],
            ],
            kf_a_init_std=self.kf_a_init_std,  # 传递 kf_a_init_std
        )

        self.tracking_armor = []
        self.last_vel_cam = np.zeros(3, dtype=float)  # 预测/融合得到的相机系速度 (m/s)

        # ========== 弹道参数（可在线调）==========
        self.use_ballistics = True           # 是否启用弹道解算（False=仅几何瞄准）
        self.bullet_speed_mps = 30.0         # 弹速 v0 (m/s) —— 可调
        self.gravity = 9.81                  # 重力加速度 (m/s^2)
        self.extra_latency_s = 0.02          # 额外延迟（视觉/通信/发射）s，用于提前量
        self.max_iters_ballistic = 3         # 弹道解算的迭代次数
        self.max_target_speed_mps = 8.0      # 目标速度限幅，抑制发散 (m/s)

    # ---------- 外部调 IMM 参数后可调用 ----------
    def rebuild_kf(self):
        dt = getattr(self.kf, "dt", 0.01)
        self.kf = MultiDimKalmanFilter(
            dt=dt,
            dim=3,
            sigma_acc=self.kf_sigma_acc,
            sigma_jerk=self.kf_sigma_jerk,
            meas_std=self.kf_meas_std,
            mu_init=(self.kf_mu_cv, self.kf_mu_ca),
            P_trans=[
                [self.kf_trans_00, self.kf_trans_01],
                [self.kf_trans_10, self.kf_trans_11],
            ],
            kf_a_init_std=self.kf_a_init_std,  # 传递 kf_a_init_std
        )

    # -------------------- KF 接口 --------------------
    def update_dt(self, dt: float):
        self.kf.set_dt(dt)

    def kf_predict(self):
        """返回下一步位置 (x,y,z)"""
        return self.kf.predict()

    def kf_update(self, x: float, y: float, z: float):
        self.kf.update(np.array([x, y, z], dtype=float))

    def get_kf_state(self):
        # 返回最近一次输出位置 (x,y,z)
        return self.kf.get_state()

    def get_kf_pos_vel(self):
        # 返回最近一次输出 (pos, vel)
        return self.kf.get_pos_vel()

    def reset_kf(self):
        self.kf.reset()
        self.last_vel_cam = np.zeros(3, dtype=float)

    # -------------------- 主入口 --------------------
    def track_armor(self, msg):
        tracking_armor = self.select_tracking_armor(msg)
        x, y, z, msg_color = self.filter(tracking_armor)

        if not self.if_find:
            yaw_sent, pitch_sent = 0.0, 0.0
            shoot_flag = 0
            return shoot_flag, yaw_sent, pitch_sent, msg_color

        # 弹道/几何解算（在枪管系求角度）
        if self.use_ballistics and self.bullet_speed_mps > 1e-3:
            yaw_sent, pitch_sent = self.solve_ballistic_angles(x, y, z)
        else:
            # 仅几何（不考虑重力与提前量）
            yaw_sent, pitch_sent = self.tf_to_gun_angles_from_cam_xyz(x, y, z)

        shoot_flag = self.if_shoot(yaw_sent, pitch_sent)
        return shoot_flag, yaw_sent, pitch_sent, msg_color

    # -------------------- 目标选择 --------------------
    def select_tracking_armor(self, msg):
        # 订阅到的是 dx,dy,dz（相机光心坐标系）
        armor_info = [Armor(a.armor_id, a.dx, a.dy, a.dz) for a in msg.armors]
        if not armor_info:
            return []

        if self.tracking_color == 1:         # 蓝色：id < 6
            filtered = [ar for ar in armor_info if ar.armor_id < 6]
        elif self.tracking_color == 0:       # 红色：id > 5
            filtered = [ar for ar in armor_info if ar.armor_id > 5]
        else:
            return []

        if not filtered:
            return []

        if len(filtered) == 1:
            return filtered

        # 取 z 最大的两个，若接近则选 |x| 更小的
        top_two = sorted(filtered, key=lambda ar: ar.z, reverse=True)[:2]
        if (top_two[0].z - top_two[1].z) <= self.track_deep_tol:
            choose = top_two[0] if abs(top_two[0].x) < abs(top_two[1].x) else top_two[1]
            return [choose]
        return [top_two[0]]

    # -------------------- KF 融合/预测 --------------------
    def filter(self, tracking_armor):
        self.tracking_armor = tracking_armor

        if not self.tracking_armor:
            if self.use_kf:
                self.lost += 1
                if self.lost <= self.frame_add and self.predict:
                    # 短时遮挡：使用一步预测
                    x, y, z = self.kf_predict()
                    # 预测态的速度
                    _, vel = self.get_kf_pos_vel()
                    self.last_vel_cam = np.array(vel, dtype=float)
                    self.if_find = True
                else:
                    self.reset_kf()
                    x, y, z = 0.0, 0.0, 0.0
                    self.lost = 0
                    self.predict = False
                    self.if_find = False
            else:
                x, y, z = 0.0, 0.0, 0.0
                self.if_find = False
        else:
            self.predict = True
            self.if_find = True
            armor = self.tracking_armor[0]
            # 相机系：把 mm 转 m
            x, y, z = armor.x / 1000.0, armor.y / 1000.0, armor.z / 1000.0
            if self.use_kf:
                self.lost = 0
                # 先融合当前观测（得到后验），再做一步预测作为输出
                self.kf_update(x, y, z)
                x, y, z = self.kf_predict()
                _, vel = self.get_kf_pos_vel()
                self.last_vel_cam = np.array(vel, dtype=float)
            else:
                self.last_vel_cam = np.zeros(3, dtype=float)

        # 打印颜色
        if self.tracking_color == 0:
            msg_color = f"追踪{self.color.RED}红色{self.color.RESET}装甲板"
        elif self.tracking_color == 1:
            msg_color = f"追踪{self.color.BLUE}蓝色{self.color.RESET}装甲板"
        else:
            msg_color = "追踪未知颜色装甲板"

        return float(x), float(y), float(z), msg_color  # 修复：返回 4 个值

    # -------------------- 相机 xyz → 枪管 yaw/pitch（几何） --------------------
    def tf_to_gun_angles_from_cam_xyz(self, x_cam_m: float, y_cam_m: float, z_cam_m: float):
        P_gun, _ = self.transform_cam_to_gun(np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float), None)
        X_gun, Y_gun, Z_gun = P_gun.tolist()
        if abs(Z_gun) < EPS:
            Z_gun = EPS
        yaw_deg = math.atan2(X_gun, Z_gun) * RAD2DEG
        pitch_deg = math.atan2(Y_gun, math.hypot(X_gun, Z_gun)) * RAD2DEG
        return yaw_deg, pitch_deg

    # -------------------- 相机位置/速度 -> 枪管系 --------------------
    def _rotation_cam_to_gun(self):
        roll = self.camera_roll_deg * DEG2RAD
        pitch = self.camera_pitch_deg * DEG2RAD
        yaw = self.camera_yaw_deg * DEG2RAD

        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll),  math.cos(roll)]])
        Ry = np.array([[ math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
        Rz = np.array([[ math.cos(yaw), -math.sin(yaw), 0],
                       [ math.sin(yaw),  math.cos(yaw), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    def transform_cam_to_gun(self, P_cam: np.ndarray, V_cam: np.ndarray | None):
        """
        P_cam: (3,) 相机系位置（m）
        V_cam: (3,) 相机系速度（m/s）或 None
        返回：P_gun, V_gun（V 为 None 则返回 None）
        """
        R = self._rotation_cam_to_gun()
        T = np.array([self.camera_tx_m, self.camera_ty_m, self.camera_tz_m], dtype=float)

        P_gun = R @ P_cam + T
        V_gun = None if V_cam is None else R @ V_cam
        return P_gun, V_gun

    # -------------------- 弹道解算（含提前量/重力） --------------------
    def solve_ballistic_angles(self, x_cam_m: float, y_cam_m: float, z_cam_m: float):
        """
        输入为相机系融合后（预测一步）的目标位置，内部会取 KF 的速度，
        统一变换到枪管系后做弹道解算，输出 yaw/pitch（deg）。
        """
        v0 = max(self.bullet_speed_mps, 1e-3)
        g = float(self.gravity)

        # 相机 -> 枪管（位置+速度）
        P_cam = np.array([x_cam_m, y_cam_m, z_cam_m], dtype=float)
        V_cam = np.array(self.last_vel_cam, dtype=float)

        # 速度限幅，避免发散（可根据运动场景再调）
        v_norm = np.linalg.norm(V_cam)
        if v_norm > self.max_target_speed_mps:
            V_cam *= (self.max_target_speed_mps / (v_norm + EPS))

        P_gun, V_gun = self.transform_cam_to_gun(P_cam, V_cam)

        # 初始 TOF 估计（无角度/无重力）
        dist0 = float(np.linalg.norm(P_gun))
        t = dist0 / v0

        # 迭代：用上一步 TOF 计算提前量 -> 重新解角度 -> 更新 TOF
        # yaw 由水平面(X,Z)的提前位置决定；pitch 用抛体低解（优先小仰角）
        yaw_deg = 0.0
        pitch_deg = 0.0

        for _ in range(self.max_iters_ballistic):
            # 预计发弹/时滞的总提前
            t_lead = max(0.0, t + float(self.extra_latency_s))

            # 目标在枪管系的“提前位置”
            P_lead = P_gun + (V_gun if V_gun is not None else 0.0) * t_lead
            X, Y, Z = float(P_lead[0]), float(P_lead[1]), float(P_lead[2])

            # 水平距离与高度差
            R_h = math.hypot(X, Z) + EPS
            H = Y

            # ---- pitch（解抛体方程，优先低弹道）----
            # tanθ = (v0^2 ± sqrt(v0^4 - g(g R^2 + 2 H v0^2))) / (g R)
            disc = v0**4 - g * (g * R_h**2 + 2.0 * H * v0**2)
            if disc >= 0.0:
                sqrt_disc = math.sqrt(disc)
                tan_theta = (v0**2 - sqrt_disc) / (g * R_h + EPS)  # 低解
                theta = math.atan(tan_theta)
            else:
                # 太远/太高导致无解：退化为几何仰角做补偿（更稳）
                theta = math.atan2(H, R_h)

            # ---- yaw（水平提前）----
            yaw = math.atan2(X, Z)

            # 更新 TOF
            t = R_h / max(EPS, v0 * math.cos(theta))

            yaw_deg = yaw * RAD2DEG
            pitch_deg = theta * RAD2DEG

        # 角度规范到 [-180,180]
        if yaw_deg > 180.0:
            yaw_deg -= 360.0
        if yaw_deg < -180.0:
            yaw_deg += 360.0

        return float(yaw_deg), float(pitch_deg)

    # -------------------- 射击判定 --------------------
    def if_shoot(self, yaw: float, pitch: float) -> int:
        return int(abs(yaw) <= self.shoot_yaw_max and
                   abs(pitch) <= self.shoot_pitch_max and
                   self.if_find)