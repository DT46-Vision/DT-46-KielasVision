import numpy as np


# =========================
# 基础：3D CV / 3D CA 子滤波器
# =========================
class _BaseKF3D:
    """3D Kalman Filter 基类，观测仅为位置 z=[x,y,z]^T。"""
    def __init__(self, dim_x: int, dt: float, meas_std: float, kf_a_init_std: float = 3.0):
        self.dim_x = dim_x
        self.dt = float(dt)
        self.meas_std = float(meas_std)
        self.kf_a_init_std = float(kf_a_init_std)  # 新增：初始加速度标准差

        # 状态与协方差
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x) * 1.0

        # 观测矩阵：只量测位置 (前3维)
        self.H = np.zeros((3, dim_x))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # 观测噪声
        self.R = np.eye(3) * (self.meas_std ** 2)

        # 状态转移/过程噪声（由子类构建）
        self.F = np.eye(dim_x)
        self.Q = np.eye(dim_x) * 1e-6  # 占位，子类重建

    # ---- 公共 API ----
    def set_dt(self, dt: float):
        self.dt = float(dt)
        self._rebuild_mats()

    def set_measurement_noise(self, meas_std: float):
        self.meas_std = float(meas_std)
        self.R = np.eye(3) * (self.meas_std ** 2)

    def set_kf_a_init_std(self, kf_a_init_std: float):
        self.kf_a_init_std = float(kf_a_init_std)

    def set_state(self, x: np.ndarray, P: np.ndarray):
        self.x = np.array(x, dtype=float).reshape((self.dim_x, 1))
        self.P = np.array(P, dtype=float).reshape((self.dim_x, self.dim_x))

    def init_from_measurement(self, z: np.ndarray, p0: float = 10.0):
        """用位置观测初始化状态（速度/加速度置0，协方差中等）"""
        z = np.array(z, dtype=float).reshape((3, 1))
        self.x[:] = 0.0
        self.x[0:3, 0] = z.flatten()
        self.P = np.eye(self.dim_x) * p0

    def predict(self, inplace: bool = True):
        """时间更新：x=F x, P=F P F^T + Q；若 inplace=False，返回预测副本，不改内部状态。"""
        if inplace:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x, self.P
        else:
            x = self.F @ self.x
            P = self.F @ self.P @ self.F.T + self.Q
            return x, P

    def innovation(self, z: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray):
        """计算创新 v, S（基于给定的预测态）"""
        z = np.array(z, dtype=float).reshape((3, 1))
        v = z - (self.H @ x_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        return v, S

    def update(self, z: np.ndarray, x_pred: np.ndarray = None, P_pred: np.ndarray = None):
        """
        量测更新：可传入预测态 (x_pred,P_pred)；若不传，则使用内部当前态（通常先 predict 再 update）。
        返回更新后的 (x, P, v, S)。
        """
        if x_pred is None or P_pred is None:
            x_pred, P_pred = self.x, self.P

        v, S = self.innovation(z, x_pred, P_pred)
        # 稳定求逆
        S_inv = np.linalg.inv(S)
        K = P_pred @ self.H.T @ S_inv
        x_new = x_pred + K @ v
        I = np.eye(self.dim_x)
        P_new = (I - K @ self.H) @ P_pred
        self.x, self.P = x_new, P_new
        return x_new, P_new, v, S

    # 子类需实现
    def _rebuild_mats(self):
        raise NotImplementedError


class CVKalman3D(_BaseKF3D):
    """
    3D 恒速 (CV) 模型
    状态：x=[px,py,pz, vx,vy,vz]^T  (6)
    过程噪声：白噪声加速度 (sigma_acc^2)
    """
    def __init__(self, dt: float, sigma_acc: float, meas_std: float, kf_a_init_std: float = 3.0):
        self.sigma_acc = float(sigma_acc)
        super().__init__(dim_x=6, dt=dt, meas_std=meas_std, kf_a_init_std=kf_a_init_std)
        self._rebuild_mats()

    def set_sigma_acc(self, sigma_acc: float):
        self.sigma_acc = float(sigma_acc)
        self._rebuild_mats()

    def _rebuild_mats(self):
        dt = self.dt
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # F
        self.F = np.block([
            [I3, dt * I3],
            [Z3, I3]
        ])

        # Q（CV: 加速度白噪声）
        q = self.sigma_acc ** 2
        Q11 = (dt ** 4) / 4.0 * q * I3
        Q12 = (dt ** 3) / 2.0 * q * I3
        Q21 = Q12
        Q22 = (dt ** 2) * q * I3
        self.Q = np.block([
            [Q11, Q12],
            [Q21, Q22]
        ])


class CAKalman3D(_BaseKF3D):
    """
    3D 恒加速度 (CA) 模型
    状态：x=[px,py,pz, vx,vy,vz, ax,ay,az]^T  (9)
    过程噪声：白噪声 jerk (sigma_jerk^2)
    """
    def __init__(self, dt: float, sigma_jerk: float, meas_std: float, kf_a_init_std: float = 3.0):
        self.sigma_jerk = float(sigma_jerk)
        super().__init__(dim_x=9, dt=dt, meas_std=meas_std, kf_a_init_std=kf_a_init_std)
        self._rebuild_mats()

    def set_sigma_jerk(self, sigma_jerk: float):
        self.sigma_jerk = float(sigma_jerk)
        self._rebuild_mats()

    def init_from_measurement(self, z: np.ndarray, p0: float = 10.0):
        """用位置观测初始化状态（速度/加速度置0，位置协方差 p0，加速度协方差 kf_a_init_std^2）"""
        z = np.array(z, dtype=float).reshape((3, 1))
        self.x[:] = 0.0
        self.x[0:3, 0] = z.flatten()
        self.P = np.eye(self.dim_x) * p0
        self.P[6:9, 6:9] = np.eye(3) * (self.kf_a_init_std ** 2)  # 加速度部分协方差

    def _rebuild_mats(self):
        dt = self.dt
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # F
        F11 = I3
        F12 = dt * I3
        F13 = 0.5 * (dt ** 2) * I3
        F21 = Z3
        F22 = I3
        F23 = dt * I3
        F31 = Z3
        F32 = Z3
        F33 = I3
        self.F = np.block([
            [F11, F12, F13],
            [F21, F22, F23],
            [F31, F32, F33],
        ])

        # Q（CA: jerk 白噪声）
        q = self.sigma_jerk ** 2
        Q11 = (dt ** 5) / 20.0 * q * I3
        Q12 = (dt ** 4) / 8.0  * q * I3
        Q13 = (dt ** 3) / 6.0  * q * I3
        Q21 = Q12
        Q22 = (dt ** 3) / 3.0  * q * I3
        Q23 = (dt ** 2) / 2.0  * q * I3
        Q31 = Q13
        Q32 = Q23
        Q33 = dt * q * I3
        self.Q = np.block([
            [Q11, Q12, Q13],
            [Q21, Q22, Q23],
            [Q31, Q32, Q33],
        ])


# =========================
# IMM (CV + CA)
# =========================
class MultiDimKalmanFilter:
    """
    3D 多模型 IMM Kalman（CV + CA）
    观测：z = [x, y, z]^T
    对外接口（兼容旧版）：
      - set_dt(dt)
      - predict()        -> 返回预测位置 (x,y,z)
      - update(z)        -> 融合观测（z 长度=3）
      - get_state()      -> 返回最近一次输出的位置
      - get_velocity()   -> 返回最近一次输出的速度
      - get_pos_vel()    -> 返回 (pos, vel)
      - reset()

    调参接口：
      - set_params(sigma_acc, sigma_jerk, meas_std, mu_init, P_trans)
      - set_transition_matrix(P)
      - set_model_probabilities(mu)
      - set_measurement_noise(std)
    """
    def __init__(
        self,
        dt: float = 0.01,
        dim: int = 3,
        sigma_acc: float = 2.0,
        sigma_jerk: float = 20.0,
        meas_std: float = 0.01,
        mu_init=(0.5, 0.5),
        P_trans=None,
        kf_a_init_std: float = 3.0,
    ):
        if dim != 3:
            raise ValueError("IMM 版本目前仅支持 dim=3（x,y,z）")

        self.dt = float(dt)
        self.dim = 3
        self.kf_a_init_std = float(kf_a_init_std)

        # 两个子滤波器
        self.cv = CVKalman3D(dt=self.dt, sigma_acc=sigma_acc, meas_std=meas_std, kf_a_init_std=kf_a_init_std)
        self.ca = CAKalman3D(dt=self.dt, sigma_jerk=sigma_jerk, meas_std=meas_std, kf_a_init_std=kf_a_init_std)

        # 模型概率
        mu_init = np.array(mu_init, dtype=float).flatten()
        if mu_init.shape != (2,):
            raise ValueError("mu_init 必须是长度为2的序列 (mu_cv, mu_ca)")
        self.mu = self._normalize_probs(mu_init)

        # 模型转移矩阵
        if P_trans is None:
            self.P_trans = np.array([[0.95, 0.05],
                                     [0.05, 0.95]], dtype=float)
        else:
            self.P_trans = np.array(P_trans, dtype=float).reshape((2, 2))

        # 最近一次输出（位置/速度）
        self._x_out = np.zeros((3, 1))
        self._v_out = np.zeros((3, 1))

        # 初始化标志
        self._initialized = False

    # ---------- 兼容旧接口 ----------
    def set_dt(self, dt: float):
        self.dt = float(dt)
        self.cv.set_dt(self.dt)
        self.ca.set_dt(self.dt)

    def set_kf_a_init_std(self, kf_a_init_std: float):
        self.kf_a_init_std = float(kf_a_init_std)
        self.ca.set_kf_a_init_std(kf_a_init_std)

    def predict(self):
        """
        对外 predict：给出“下一步”的融合位置预测，但不推进内部后验。
        同时返回/保存融合速度的预测值。
        """
        if not self._initialized:
            return self._x_out.flatten()

        # 复制后预测（不改内部后验）
        x_cv_pred, _ = self.cv.predict(inplace=False)  # (6,1)
        x_ca_pred, _ = self.ca.predict(inplace=False)  # (9,1)

        # 融合位置/速度（只取前6维里的位置+速度）
        pos = self.mu[0] * x_cv_pred[0:3, :] + self.mu[1] * x_ca_pred[0:3, :]
        vel = self.mu[0] * x_cv_pred[3:6, :] + self.mu[1] * x_ca_pred[3:6, :]

        self._x_out = pos
        self._v_out = vel
        return self._x_out.flatten()

    def update(self, z):
        """
        IMM 标准步骤（对当前时刻）：
          1) 交互混合（带维度映射）
          2) 各模型时间更新（predict）
          3) 各模型量测更新（update）并计算似然
          4) 更新模型概率 mu
          5) 融合后验 (x_fused, v_fused)
        """
        z = np.array(z, dtype=float).reshape((3, 1))

        # 首帧：用观测初始化
        if not self._initialized:
            self.cv.init_from_measurement(z, p0=10.0)
            self.ca.init_from_measurement(z, p0=10.0)
            self._x_out = z.copy()
            self._v_out = np.zeros((3, 1))
            self._initialized = True
            return

        # === 1) 交互混合 ===
        c = self.P_trans.T @ self.mu             # (2,)
        c = np.clip(c, 1e-8, None)
        mu_cond = (self.P_trans * self.mu.reshape((2, 1))) / c.reshape((1, 2))

        x_cv, P_cv = self.cv.x, self.cv.P        # (6,1), (6,6)
        x_ca, P_ca = self.ca.x, self.ca.P        # (9,1), (9,9)

        # --- 维度映射矩阵 ---
        H_ca2cv = np.zeros((6, 9))
        H_ca2cv[0, 0] = 1.0
        H_ca2cv[1, 1] = 1.0
        H_ca2cv[2, 2] = 1.0
        H_ca2cv[3, 3] = 1.0
        H_ca2cv[4, 4] = 1.0
        H_ca2cv[5, 5] = 1.0

        G_cv2ca = np.zeros((9, 6))
        G_cv2ca[0:6, 0:6] = np.eye(6)

        Qa = np.zeros((9, 9))
        Qa[6:9, 6:9] = np.eye(3) * 1e2

        # --- 混合均值 ---
        x0_cv = mu_cond[0, 0] * x_cv + mu_cond[1, 0] * (H_ca2cv @ x_ca)   # (6,1)
        x0_ca = mu_cond[0, 1] * (G_cv2ca @ x_cv) + mu_cond[1, 1] * x_ca   # (9,1)

        # --- 混合协方差 ---
        dx_cv     = x_cv - x0_cv
        dx_ca2cv  = (H_ca2cv @ x_ca) - x0_cv
        P0_cv = (
            mu_cond[0, 0] * (P_cv + dx_cv @ dx_cv.T) +
            mu_cond[1, 0] * (H_ca2cv @ P_ca @ H_ca2cv.T + dx_ca2cv @ dx_ca2cv.T)
        )

        dx_ca     = x_ca - x0_ca
        dx_cv2ca  = (G_cv2ca @ x_cv) - x0_ca
        P0_ca = (
            mu_cond[0, 1] * (G_cv2ca @ P_cv @ G_cv2ca.T + Qa + dx_cv2ca @ dx_cv2ca.T) +
            mu_cond[1, 1] * (P_ca + dx_ca @ dx_ca.T)
        )

        # 作为各模型先验
        self.cv.set_state(x0_cv, P0_cv)
        self.ca.set_state(x0_ca, P0_ca)

        # === 2) 时间更新 ===
        x_cv_pred, P_cv_pred = self.cv.predict(inplace=True)
        x_ca_pred, P_ca_pred = self.ca.predict(inplace=True)

        # === 3) 量测更新 + 似然 ===
        x_cv_upd, P_cv_upd, v_cv, S_cv = self.cv.update(z, x_cv_pred, P_cv_pred)
        x_ca_upd, P_ca_upd, v_ca, S_ca = self.ca.update(z, x_ca_pred, P_ca_pred)

        L_cv = self._gaussian_likelihood(v_cv, S_cv)
        L_ca = self._gaussian_likelihood(v_ca, S_ca)

        # === 4) 更新 mu ===
        c_trans = self.P_trans.T @ self.mu
        numer = np.array([L_cv * c_trans[0], L_ca * c_trans[1]], dtype=float)
        denom = np.sum(numer) + 1e-12
        self.mu = numer / denom

        # === 5) 融合后验（保存位置+速度）===
        self.cv.x, self.cv.P = x_cv_upd, P_cv_upd
        self.ca.x, self.ca.P = x_ca_upd, P_ca_upd

        x_fused_pos = self.mu[0] * self.cv.x[0:3, :] + self.mu[1] * self.ca.x[0:3, :]
        v_fused     = self.mu[0] * self.cv.x[3:6, :] + self.mu[1] * self.ca.x[3:6, :]

        self._x_out = x_fused_pos
        self._v_out = v_fused

    def get_state(self):
        """返回最近一次 predict/update 的输出位置 (x,y,z)。"""
        return self._x_out.flatten()

    def get_velocity(self):
        """返回最近一次 predict/update 的输出速度 (vx,vy,vz)。"""
        return self._v_out.flatten()

    def get_pos_vel(self):
        """返回 (pos, vel)，均为 1D ndarray 长度3。"""
        return self._x_out.flatten(), self._v_out.flatten()

    def reset(self):
        self.cv = CVKalman3D(dt=self.dt, sigma_acc=self.cv.sigma_acc, meas_std=self.cv.meas_std, kf_a_init_std=self.kf_a_init_std)
        self.ca = CAKalman3D(dt=self.dt, sigma_jerk=self.ca.sigma_jerk, meas_std=self.ca.meas_std, kf_a_init_std=self.kf_a_init_std)
        self.mu = self._normalize_probs(self.mu)
        self._x_out = np.zeros((3, 1))
        self._v_out = np.zeros((3, 1))
        self._initialized = False

    # ---------- 调参接口 ----------
    def set_params(self, sigma_acc=None, sigma_jerk=None, meas_std=None, mu_init=None, P_trans=None, kf_a_init_std=None):
        """一次性设置若干参数；未传的保持不变。"""
        if sigma_acc is not None:
            self.cv.set_sigma_acc(float(sigma_acc))
        if sigma_jerk is not None:
            self.ca.set_sigma_jerk(float(sigma_jerk))
        if meas_std is not None:
            self.cv.set_measurement_noise(float(meas_std))
            self.ca.set_measurement_noise(float(meas_std))
        if mu_init is not None:
            self.set_model_probabilities(mu_init)
        if P_trans is not None:
            self.set_transition_matrix(P_trans)
        if kf_a_init_std is not None:
            self.set_kf_a_init_std(kf_a_init_std)

    def set_transition_matrix(self, P):
        P = np.array(P, dtype=float).reshape((2, 2))
        row_sums = P.sum(axis=1, keepdims=True) + 1e-12  # 保证行归一
        self.P_trans = P / row_sums

    def set_model_probabilities(self, mu):
        mu = np.array(mu, dtype=float).flatten()
        if mu.shape != (2,):
            raise ValueError("mu 必须是长度为2的序列")
        self.mu = self._normalize_probs(mu)

    def set_measurement_noise(self, meas_std):
        self.cv.set_measurement_noise(meas_std)
        self.ca.set_measurement_noise(meas_std)

    # ---------- 工具 ----------
    @staticmethod
    def _normalize_probs(mu):
        mu = np.maximum(mu, 1e-12)
        mu = mu / np.sum(mu)
        return mu

    @staticmethod
    def _gaussian_likelihood(v, S):
        """多元高斯密度 N(v;0,S) 的值（非对数），用于模型似然。"""
        k = v.shape[0]  # 3
        try:
            detS = np.linalg.det(S)
            if detS <= 0:
                return 1e-12
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 1e-12
        exponent = float(-0.5 * (v.T @ invS @ v))
        norm = float(((2.0 * np.pi) ** (-k / 2.0)) * (detS ** -0.5))
        return max(1e-12, norm * np.exp(exponent))