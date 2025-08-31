"""
Armor Tracker Node (IMM-3D, xyz 输入=米)
--------------------------------------
- 订阅:  /detector/armors_info (ArmorsCppMsg) —— armors[i].dx,dy,dz 单位: **米**
- 转换:  本节点将 dx,dy,dz 由米 -> 毫米，再交给 Tracker (IMM-3D)
- 追踪:  交给 rm_tracker.armor_tracker.Tracker（内部 IMM 基于 xyz(mm)）
- 发布:  /tracker/target (ArmorTracking) —— yaw/pitch(度) 与 shoot_flag
- 参数:  IMM + 弹道参数均暴露在 ROS 参数中，改动会即时生效
"""

from types import SimpleNamespace
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from rcl_interfaces.msg import SetParametersResult

from rm_interfaces.msg import ArmorsCppMsg, ArmorTracking, Decision
from rm_tracker.armor_tracker import Tracker


class LogThrottler:
    def __init__(self, node: Node, default_ms: int = 1000):
        self._node = node
        self._default_ms = int(default_ms)
        self._last_ns = {}

    def should_log(self, key: str, throttle_ms: int = None) -> bool:
        if throttle_ms is None:
            throttle_ms = self._default_ms
        throttle_ns = int(throttle_ms) * 1_000_000
        now_ns = self._node.get_clock().now().nanoseconds
        last_ns = self._last_ns.get(key)
        if last_ns is None or (now_ns - last_ns) >= throttle_ns:
            self._last_ns[key] = now_ns
            return True
        return False


class ArmorTrackerNode(Node):
    def __init__(self, name: str = "armor_tracker_node"):
        super().__init__(name)

        # Tracker（内部基于 mm）
        self.tracker = Tracker(logger=self.get_logger())
        self._c = self.tracker.color

        # ---------- 打印/FPS ----------
        self.declare_parameter('log_throttle_ms', 1000)
        self.declare_parameter('fps_window_sec', 1.0)

        # ---------- 追踪参数 ----------
        self.declare_parameter('use_kf',           True)
        self.declare_parameter('frame_add',        60)
        self.declare_parameter('tracking_color',   0)     # 0:红, 1:蓝, 10:不追
        self.declare_parameter('follow_decision',  0)
        self.declare_parameter('track_deep_tol',   10.0)  # mm
        self.declare_parameter('shoot_yaw_max',    1.5)   # deg
        self.declare_parameter('shoot_pitch_max',  1.0)   # deg

        # ---------- 外参（米/度） ----------
        self.declare_parameter('camera_tx_m',      0.0)
        self.declare_parameter('camera_ty_m',      0.0)
        self.declare_parameter('camera_tz_m',     -0.03)
        self.declare_parameter('camera_yaw_deg',   4.0)
        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('camera_roll_deg',  0.0)

        # ---------- IMM 调参 ----------
        self.declare_parameter('kf_sigma_acc',     2.0)
        self.declare_parameter('kf_sigma_jerk',    20.0)
        self.declare_parameter('kf_meas_std',      0.01)
        self.declare_parameter('kf_mu_cv',         0.6)
        self.declare_parameter('kf_mu_ca',         0.4)
        self.declare_parameter('kf_trans_00',      0.9)
        self.declare_parameter('kf_trans_01',      0.1)
        self.declare_parameter('kf_trans_10',      0.1)
        self.declare_parameter('kf_trans_11',      0.9)
        self.declare_parameter('kf_a_init_std',    3.0)   # 新增：初始加速度标准差

        # ---------- 弹道参数 ----------
        self.declare_parameter('use_ballistics', True)
        self.declare_parameter('bullet_speed_mps', 30.0)   # 默认弹速
        self.declare_parameter('gravity', 9.81)            # 重力加速度
        self.declare_parameter('extra_latency_s', 0.02)    # 控制延迟补偿

        self.add_on_set_parameters_callback(self._on_params)

        # ---------- 通信 ----------
        self.sub_armors = self.create_subscription(
            ArmorsCppMsg, '/detector/armors_info', self._cb_armors, 10)
        self.sub_decision = self.create_subscription(
            Decision, '/nav/decision', self._cb_decision, 10)
        self.pub_tracking = self.create_publisher(ArmorTracking, '/tracker/target', 10)

        # ---------- 计时/FPS ----------
        self.log = LogThrottler(self, default_ms=int(self.get_parameter('log_throttle_ms').value))
        self._fps_window_sec = float(self.get_parameter('fps_window_sec').value)
        self._last_fps_time = self.get_clock().now()
        self._processed_in_window = 0

        # dt 用于给 Tracker/IMM
        self._last_update_time = self.get_clock().now()

        self.get_logger().info('Armor Tracker Node started (IMM-3D + Ballistics).')

        # 初始化：同步参数到 tracker + 重建 KF
        self._sync_all_params_to_tracker(rebuild=True)

    # ---------- 同步全部参数到 Tracker ----------
    def _sync_all_params_to_tracker(self, rebuild=False):
        t = self.tracker
        gp = self.get_parameter

        # 追踪与限幅
        t.use_kf          = bool(gp('use_kf').value)
        t.frame_add       = int(gp('frame_add').value)
        t.tracking_color  = int(gp('tracking_color').value)
        t.follow_decision = int(gp('follow_decision').value)
        t.track_deep_tol  = float(gp('track_deep_tol').value)
        t.shoot_yaw_max   = float(gp('shoot_yaw_max').value)
        t.shoot_pitch_max = float(gp('shoot_pitch_max').value)

        # 外参（米 -> 毫米转换交给 Tracker 内部）
        t.camera_tx_m      = float(gp('camera_tx_m').value)
        t.camera_ty_m      = float(gp('camera_ty_m').value)
        t.camera_tz_m      = float(gp('camera_tz_m').value)
        t.camera_yaw_deg   = float(gp('camera_yaw_deg').value)
        t.camera_pitch_deg = float(gp('camera_pitch_deg').value)
        t.camera_roll_deg  = float(gp('camera_roll_deg').value)

        # IMM 参数
        t.kf_sigma_acc   = float(gp('kf_sigma_acc').value)
        t.kf_sigma_jerk  = float(gp('kf_sigma_jerk').value)
        t.kf_meas_std    = float(gp('kf_meas_std').value)
        t.kf_mu_cv       = float(gp('kf_mu_cv').value)
        t.kf_mu_ca       = float(gp('kf_mu_ca').value)
        t.kf_trans_00    = float(gp('kf_trans_00').value)
        t.kf_trans_01    = float(gp('kf_trans_01').value)
        t.kf_trans_10    = float(gp('kf_trans_10').value)
        t.kf_trans_11    = float(gp('kf_trans_11').value)
        t.kf_a_init_std  = float(gp('kf_a_init_std').value)  # 新增：同步 kf_a_init_std

        # 弹道参数
        t.use_ballistics   = bool(gp('use_ballistics').value)
        t.bullet_speed_mps = float(gp('bullet_speed_mps').value)
        t.gravity          = float(gp('gravity').value)
        t.extra_latency_s  = float(gp('extra_latency_s').value)

        if rebuild:
            try:
                t.rebuild_kf()
                self.get_logger().warn("IMM 参数同步完成：已重建滤波器")
            except Exception as e:
                self.get_logger().error(f"重建 IMM 失败：{e}")

    # ---------- 动态参数回调 ----------
    def _on_params(self, params):
        need_rebuild = False
        for p in params:
            name, val = p.name, p.value
            try:
                if name == 'log_throttle_ms':
                    self.log._default_ms = int(val)
                elif name == 'fps_window_sec':
                    self._fps_window_sec = float(val)
                elif name == 'kf_a_init_std':  # 新增：处理 kf_a_init_std
                    self.tracker.kf_a_init_std = float(val)
                    need_rebuild = True
                elif hasattr(self.tracker, name):
                    setattr(self.tracker, name, val)
                    if name.startswith('kf_'):
                        need_rebuild = True
            except Exception as e:
                self.get_logger().warn(f"参数 {name} 更新失败：{e}")

        if need_rebuild:
            try:
                self.tracker.rebuild_kf()
                self.get_logger().warn("IMM 参数更新：已重建滤波器")
            except Exception as e:
                self.get_logger().error(f"重建 IMM 失败：{e}")

        return SetParametersResult(successful=True)

    # ---------- Armors 回调 ----------
    def _cb_armors(self, msg: ArmorsCppMsg):
        try:
            # 更新时间步 dt
            now = self.get_clock().now()
            dt = (now - self._last_update_time).nanoseconds / 1e9
            if dt <= 0.0 or dt > 0.2:
                dt = 0.01
            self._last_update_time = now

            if hasattr(self.tracker, 'update_dt'):
                self.tracker.update_dt(dt)

            # 不追踪模式
            track_color = int(self.get_parameter('tracking_color').value)
            if track_color == 10:
                if self.log.should_log('no_track_color'):
                    self.get_logger().warn("tracking_color=10：不追踪装甲板。")
                return

            # 调 Tracker
            shoot_flag, yaw_sent, pitch_sent, msg_color = self.tracker.track_armor(msg)

            # 发布
            out = ArmorTracking()
            out.header = Header()
            out.header.stamp = now.to_msg()
            out.header.frame_id = 'tracking_armor_frame'
            out.yaw = float(yaw_sent)
            out.pitch = float(pitch_sent)
            out.shoot_flag = int(shoot_flag)
            self.pub_tracking.publish(out)

            # 日志
            if msg_color and self.log.should_log('track_color'):
                self.get_logger().info(msg_color)

            if self.log.should_log('pub_tracking'):
                c = self._c
                self.get_logger().info(
                    f"发布 {c.CYAN}yaw{c.RESET}:{c.PINK}{out.yaw:.2f}{c.RESET} "
                    f"|| {c.CYAN}pitch{c.RESET}:{c.PINK}{out.pitch:.2f}{c.RESET} "
                    f"|| {c.CYAN}shoot{c.RESET}:{c.PINK}{out.shoot_flag:.0f}{c.RESET}"
                )

            # FPS
            self._processed_in_window += 1
            if (now - self._last_fps_time).nanoseconds >= int(self._fps_window_sec * 1e9):
                secs = (now - self._last_fps_time).nanoseconds / 1e9
                fps = self._processed_in_window / max(secs, 1e-6)
                self.get_logger().warn(f"[FPS_track] {fps:.1f}")
                self._processed_in_window = 0
                self._last_fps_time = now

        except Exception as e:
            self.get_logger().error(f"处理 ArmorsCppMsg 异常：{e}")

    # ---------- 决策回调 ----------
    def _cb_decision(self, msg: Decision):
        try:
            follow = int(self.get_parameter('follow_decision').value)
            if follow != 1:
                return
            if hasattr(msg, 'color'):
                cur = int(self.get_parameter('tracking_color').value)
                if msg.color != cur:
                    self.set_parameters([rclpy.parameter.Parameter(
                        'tracking_color',
                        rclpy.parameter.Parameter.Type.INTEGER,
                        int(msg.color)
                    )])
                    self.get_logger().warn(f"追踪颜色切换为 {int(msg.color)}")
        except Exception as e:
            self.get_logger().error(f"处理 Decision 异常：{e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArmorTrackerNode("armor_tracker_node")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()