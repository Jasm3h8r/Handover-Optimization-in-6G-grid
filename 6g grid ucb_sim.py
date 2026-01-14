import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

# Custom Gaussian smoothing filter
def gaussian_filter(array, sigma=3):
    """Simple Gaussian smoothing using numpy convolution"""
    size = int(sigma * 4)
    x = np.arange(-size, size + 1)
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    result = array.copy()
    for i in range(result.shape[0]):
        result[i, :] = np.convolve(result[i, :], kernel_1d, mode='same')
    for j in range(result.shape[1]):
        result[:, j] = np.convolve(result[:, j], kernel_1d, mode='same')
    
    return result

# 6G FREQUENCY BANDS (GHz)
BAND_6G = {
    # Sub-6 (backward compatibility)
    'sub6': {'freq': 3.5, 'bw': 100},
    # mmWave (5G carryover)
    'mmwave_low': {'freq': 28, 'bw': 400},
    'mmwave_high': {'freq': 47, 'bw': 800},
    # Sub-THz (6G new)
    'subthz_low': {'freq': 95, 'bw': 2000},
    'subthz_mid': {'freq': 140, 'bw': 5000}
}

# 6G PARAMETERS
MAX_SPECTRAL_EFFICIENCY_6G = 12.0  # bits/s/Hz (6G with advanced modulation)
MIN_SINR_DB = -10.0
MAX_SINR_DB = 35.0  # Slightly higher for 6G

# Sub-THz molecular absorption coefficients (dB/km)
SUBTHZ_ABSORPTION = {
    95: 0.5,
    140: 2.0
}

@dataclass
class BaseStation6G:
    id: int
    x: float
    y: float
    z: float  # 3D coordinate
    freq: float
    power: float
    beam_direction: float
    elevation_angle: float  # 3D beamforming
    bs_type: str  # 'macro', 'micro', 'pico'
    
    # Massive MIMO parameters (6G scale)
    num_antennas: int = 256
    antenna_spacing: float = 0.5
    max_simultaneous_beams: int = 64
    beamforming_gain_db: float = 30.0
    
    # Bandwidth (MHz)
    bandwidth_mhz: float = 2000.0
    
    # State
    load: int = 0
    active_beams: Dict[int, int] = field(default_factory=dict)
    beam_angles: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # (azimuth, elevation)

@dataclass
class UserEquipment6G:
    id: int
    x: float
    y: float
    z: float  # 3D coordinate (height)
    vx: float
    vy: float
    vz: float = 0.0  # Vertical velocity (for future UAV support)
    
    # UE antenna configuration
    num_antennas: int = 8
    
    # Connection state
    connected_bs: Optional[int] = None
    beam_id: Optional[int] = None
    sinr: float = 0.0
    throughput: float = 0.0
    
    # TTT for handover hysteresis
    candidate_bs: Optional[int] = None
    ttt_counter: int = 0
    ttt_threshold: int = 5
    
    # Tracking
    handover_count: int = 0
    time_since_last_handover: int = 0
    outage_count: int = 0  # 6G metric

class UCB:
    """Standard UCB (Upper Confidence Bound) Multi-Armed Bandit"""
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c  # Exploration parameter
        
        # Per-UE tracking
        self.arm_counts = {}  # ue_id -> [counts per arm]
        self.arm_rewards = {}  # ue_id -> [sum of rewards per arm]
        self.total_counts = {}  # ue_id -> total pulls
        
        self.total_reward = 0.0
        self.optimal_reward = 0.0
        self.decision_count = 0
    
    def _initialize_ue(self, ue_id: int):
        """Initialize tracking for a new UE"""
        if ue_id not in self.arm_counts:
            self.arm_counts[ue_id] = np.zeros(self.n_arms)
            self.arm_rewards[ue_id] = np.zeros(self.n_arms)
            self.total_counts[ue_id] = 0
    
    def select_arm(self, ue_id: int, current_arm: Optional[int] = None,
                   handover_penalty: float = 0.2) -> Tuple[int, List[Dict], bool]:
        """Select arm using UCB algorithm"""
        self._initialize_ue(ue_id)
        
        ucb_values = []
        best_arm = 0
        best_ucb = -np.inf
        
        # Ensure all arms are tried at least once
        for arm in range(self.n_arms):
            if self.arm_counts[ue_id][arm] == 0:
                ucb_values.append({
                    'arm': arm,
                    'ucb': float('inf'),
                    'expected_reward': 0,
                    'uncertainty': float('inf')
                })
                if best_ucb < float('inf'):
                    best_ucb = float('inf')
                    best_arm = arm
            else:
                # Calculate average reward
                avg_reward = self.arm_rewards[ue_id][arm] / self.arm_counts[ue_id][arm]
                
                # Apply handover penalty if switching arms
                if current_arm is not None and arm != current_arm:
                    avg_reward -= handover_penalty
                    hysteresis_margin = 0.15
                    avg_reward -= hysteresis_margin
                
                # Calculate exploration bonus
                exploration_bonus = self.c * np.sqrt(
                    np.log(self.total_counts[ue_id] + 1) / self.arm_counts[ue_id][arm]
                )
                
                ucb = avg_reward + exploration_bonus
                
                ucb_values.append({
                    'arm': arm,
                    'ucb': ucb,
                    'expected_reward': avg_reward,
                    'uncertainty': exploration_bonus
                })
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_arm = arm
        
        should_handover = (current_arm is None) or (best_arm != current_arm)
        return best_arm, ucb_values, should_handover
    
    def update(self, ue_id: int, arm: int, reward: float):
        """Update arm statistics with observed reward"""
        self._initialize_ue(ue_id)
        
        self.arm_counts[ue_id][arm] += 1
        self.arm_rewards[ue_id][arm] += reward
        self.total_counts[ue_id] += 1
        self.decision_count += 1
    
    def reset(self):
        """Reset all statistics"""
        self.arm_counts = {}
        self.arm_rewards = {}
        self.total_counts = {}
        self.total_reward = 0.0
        self.optimal_reward = 0.0
        self.decision_count = 0


class HetNet6GSimulator:
    """6G Ground HetNet Simulator with Sub-THz Support"""
    
    def __init__(self, grid_size: int = 3000, 
                 num_macro: int = 4, num_micro: int = 8, num_pico: int = 12,
                 num_ues: int = 1000, tx_power_macro: float = 43, 
                 noise_power: float = -174, ucb_c: float = 2.0):
        
        self.grid_size = grid_size
        self.num_macro = num_macro
        self.num_micro = num_micro
        self.num_pico = num_pico
        self.num_bs = num_macro + num_micro + num_pico
        self.num_ues = num_ues
        self.tx_power_macro = tx_power_macro
        self.noise_power = noise_power
        self.ucb_c = ucb_c
        
        # Network components
        self.base_stations: List[BaseStation6G] = []
        self.users: List[UserEquipment6G] = []
        self.ucb: UCB = None
        
        # Channel state cache
        self.channel_cache = {}
        self.shadow_fading_map = {}
        
        # Tracking
        self.step = 0
        self.logs = []
        self.step_metrics = {
            'avg_sinr': [], 'total_throughput': [], 'handover_count': [],
            'coverage': [], 'regret': [], 'avg_load': [],
            'cell_edge_rate': [], 'outage_prob': [], 'pingpong_rate': []
        }
        
        # 6G-specific metrics
        self.throughput_distribution = []
        self.sinr_samples = []
        
        self._initialize_hetnet()
        self._initialize_shadow_map()
    
    def _initialize_hetnet(self):
        """Initialize 6G HetNet"""
        self.base_stations = self._deploy_6g_hetnet()
        self.users = self._generate_users()
        self.ucb = UCB(n_arms=self.num_bs, c=self.ucb_c)
        self._add_log(f'6G HetNet initialized: {self.num_macro}M + {self.num_micro}m + '
                     f'{self.num_pico}p BSs, {self.num_ues} UEs')
    
    def _initialize_shadow_map(self):
        """Initialize spatially correlated shadow fading map"""
        grid_points = 50
        x = np.linspace(0, self.grid_size, grid_points)
        y = np.linspace(0, self.grid_size, grid_points)
        
        for bs in self.base_stations:
            # Higher variance for Sub-THz bands
            if bs.freq > 90:
                sigma = 8.0
            else:
                sigma = 4.0
            
            shadow_map = np.random.normal(0, sigma, (grid_points, grid_points))
            shadow_map = gaussian_filter(shadow_map, sigma=3)
            self.shadow_fading_map[bs.id] = {
                'x': x, 'y': y, 'values': shadow_map
            }
    
    def _get_shadow_fading(self, bs_id: int, ue_x: float, ue_y: float) -> float:
        """Get spatially correlated shadow fading value"""
        shadow_map = self.shadow_fading_map[bs_id]
        
        x_idx = int((ue_x / self.grid_size) * (len(shadow_map['x']) - 1))
        y_idx = int((ue_y / self.grid_size) * (len(shadow_map['y']) - 1))
        
        x_idx = np.clip(x_idx, 0, len(shadow_map['x']) - 1)
        y_idx = np.clip(y_idx, 0, len(shadow_map['y']) - 1)
        
        return np.clip(shadow_map['values'][y_idx, x_idx], -15, 15)
    
    def _deploy_6g_hetnet(self) -> List[BaseStation6G]:
        """Deploy 6G heterogeneous network"""
        bs_list = []
        bs_id = 0
        
        # 1. MACRO CELLS (Sub-6 / mmWave for wide coverage)
        for i in range(self.num_macro):
            angle = (2 * np.pi * i) / self.num_macro
            radius = self.grid_size / 3.5
            x = self.grid_size / 2 + radius * np.cos(angle)
            y = self.grid_size / 2 + radius * np.sin(angle)
            z = 30.0  # 30m height
            
            freq = np.random.choice([3.5, 28])
            bw = 100 if freq < 10 else 400
            
            bs_list.append(BaseStation6G(
                id=bs_id, x=x, y=y, z=z,
                freq=freq,
                power=self.tx_power_macro,
                beam_direction=np.random.uniform(0, 360),
                elevation_angle=np.random.uniform(-10, 10),
                bs_type='macro',
                num_antennas=128,
                max_simultaneous_beams=256,
                beamforming_gain_db=25.0,
                bandwidth_mhz=bw
            ))
            bs_id += 1
        
        # 2. MICRO CELLS (mmWave / Sub-THz)
        for i in range(self.num_micro):
            x = np.random.uniform(self.grid_size * 0.15, self.grid_size * 0.85)
            y = np.random.uniform(self.grid_size * 0.15, self.grid_size * 0.85)
            z = 15.0  # 15m height
            
            freq = np.random.choice([28, 47, 95])
            bw = 400 if freq < 50 else 2000
            
            bs_list.append(BaseStation6G(
                id=bs_id, x=x, y=y, z=z,
                freq=freq,
                power=self.tx_power_macro - 8,
                beam_direction=np.random.uniform(0, 360),
                elevation_angle=np.random.uniform(-15, 15),
                bs_type='micro',
                num_antennas=256,
                max_simultaneous_beams=128,
                beamforming_gain_db=30.0,
                bandwidth_mhz=bw
            ))
            bs_id += 1
        
        # 3. PICO CELLS (Sub-THz)
        for i in range(self.num_pico):
            x = np.random.uniform(0, self.grid_size)
            y = np.random.uniform(0, self.grid_size)
            z = 8.0  # 8m height
            
            freq = np.random.choice([95, 140])
            bw = 2000 if freq < 120 else 5000
            
            bs_list.append(BaseStation6G(
                id=bs_id, x=x, y=y, z=z,
                freq=freq,
                power=self.tx_power_macro - 15,
                beam_direction=np.random.uniform(0, 360),
                elevation_angle=np.random.uniform(-20, 20),
                bs_type='pico',
                num_antennas=512,
                max_simultaneous_beams=64,
                beamforming_gain_db=35.0,
                bandwidth_mhz=bw
            ))
            bs_id += 1
        
        return bs_list
    
    def _generate_users(self) -> List[UserEquipment6G]:
        """Generate mobile users with 3D coordinates"""
        return [UserEquipment6G(
            id=i,
            x=np.random.uniform(0, self.grid_size),
            y=np.random.uniform(0, self.grid_size),
            z=1.5,  # Ground UE at 1.5m height
            vx=np.random.uniform(-2, 2),
            vy=np.random.uniform(-2, 2),
            vz=0.0,
            num_antennas=np.random.choice([4, 8, 16])
        ) for i in range(self.num_ues)]
    
    def _add_log(self, message: str):
        self.logs.append(f'[S{self.step}] {message}')
        if len(self.logs) > 30:
            self.logs = self.logs[-30:]
    
    @staticmethod
    def _distance_3d(x1, y1, z1, x2, y2, z2) -> float:
        """3D Euclidean distance"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    
    @staticmethod
    def _angle_between(x1, y1, x2, y2) -> float:
        """Azimuth angle in degrees"""
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    
    @staticmethod
    def _elevation_angle(x1, y1, z1, x2, y2, z2) -> float:
        """Elevation angle in degrees"""
        horizontal_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if horizontal_dist < 0.1:
            return 0.0
        return np.arctan2(z2 - z1, horizontal_dist) * 180 / np.pi
    
    @staticmethod
    def _los_probability_6g(d: float, freq: float, bs_type: str) -> float:
        """6G LOS probability - more restrictive for Sub-THz"""
        if freq > 90:  # Sub-THz bands
            d0, d1 = 100, 40
        elif bs_type == 'macro':
            d0, d1 = 250, 150
        elif bs_type == 'micro':
            d0, d1 = 150, 80
        else:
            d0, d1 = 80, 40
        return min(d0 / max(d, 1), 1) * (1 - np.exp(-d / d1)) + np.exp(-d / d1)
    
    def _molecular_absorption_loss(self, freq: float, distance_m: float) -> float:
        """Sub-THz molecular absorption loss (water vapor, oxygen)"""
        if freq < 90:
            return 0.0
        
        # Find closest absorption coefficient
        absorption_db_per_km = 0.0
        for f, abs_coef in SUBTHZ_ABSORPTION.items():
            if abs(freq - f) < 50:
                absorption_db_per_km = abs_coef
                break
        
        distance_km = distance_m / 1000.0
        return absorption_db_per_km * distance_km
    
    def _path_loss_6g(self, d: float, fc: float, is_los: bool, z_bs: float, z_ue: float) -> float:
        """6G path loss model with 3D geometry and Sub-THz effects"""
        d = max(d, 1)
        
        # Base path loss (3GPP-like for sub-THz)
        if is_los:
            pl = 32.4 + 21 * np.log10(d) + 20 * np.log10(fc)
        else:
            pl_los = 32.4 + 21 * np.log10(d) + 20 * np.log10(fc)
            nlos_penalty = 25 + 0.6 * (fc - 24)
            pl = pl_los + nlos_penalty
        
        # Sub-THz molecular absorption
        if fc > 90:
            mol_loss = self._molecular_absorption_loss(fc, d)
            pl += mol_loss
        
        # Height-dependent correction
        height_diff = abs(z_bs - z_ue)
        if height_diff > 10:
            pl += 2 * np.log10(height_diff / 10)
        
        return pl
    
    @staticmethod
    def _rayleigh_fading() -> float:
        """Rayleigh fading for NLOS"""
        real = np.random.normal(0, 1)
        imag = np.random.normal(0, 1)
        magnitude = np.sqrt(real**2 + imag**2)
        fading_db = 20 * np.log10(magnitude / np.sqrt(2))
        return np.clip(fading_db, -10, 6)
    
    @staticmethod
    def _rician_fading(k_factor_db: float = 12.0) -> float:
        """Rician fading for LOS (higher K-factor for 6G beamforming)"""
        k_linear = 10 ** (k_factor_db / 10)
        los_component = np.sqrt(k_linear / (k_linear + 1))
        scatter_power = 1 / (k_linear + 1)
        real = np.random.normal(0, np.sqrt(scatter_power / 2))
        imag = np.random.normal(0, np.sqrt(scatter_power / 2))
        total_real = los_component + real
        total_imag = imag
        magnitude = np.sqrt(total_real**2 + total_imag**2)
        fading_db = 20 * np.log10(magnitude)
        return np.clip(fading_db, -5, 10)
    
    def _get_or_create_channel(self, user: UserEquipment6G, bs: BaseStation6G) -> Dict:
        """Get or create channel state with coherence"""
        key = (user.id, bs.id)
        
        if key not in self.channel_cache:
            d = self._distance_3d(user.x, user.y, user.z, bs.x, bs.y, bs.z)
            
            # LOS state
            los_seed = (self.step // 10) * 1000 + user.id * 100 + bs.id
            np.random.seed(los_seed)
            is_los = np.random.uniform(0, 1) < self._los_probability_6g(d, bs.freq, bs.bs_type)
            
            # Shadow fading
            shadow_fade = self._get_shadow_fading(bs.id, user.x, user.y)
            
            # Small-scale fading
            fade_seed = (self.step // 5) * 1000 + user.id * 100 + bs.id
            np.random.seed(fade_seed)
            if is_los:
                k_factor = max(8, 20 - d/50)
                small_scale_fade = self._rician_fading(k_factor_db=k_factor)
            else:
                small_scale_fade = self._rayleigh_fading()
            
            np.random.seed(None)
            
            self.channel_cache[key] = {
                'los': is_los,
                'shadow': shadow_fade,
                'small_scale': small_scale_fade
            }
        
        return self.channel_cache[key]
    
    def _calculate_beamforming_gain_3d(self, bs: BaseStation6G, user: UserEquipment6G) -> float:
        """3D beamforming gain with azimuth and elevation"""
        ue_azimuth = self._angle_between(bs.x, bs.y, user.x, user.y)
        ue_elevation = self._elevation_angle(bs.x, bs.y, bs.z, user.x, user.y, user.z)
        
        if user.id in bs.active_beams:
            beam_angles = bs.beam_angles[bs.active_beams[user.id]]
            beam_azimuth, beam_elevation = beam_angles
        else:
            beam_azimuth = ue_azimuth
            beam_elevation = ue_elevation
        
        # Azimuth alignment
        azimuth_diff = abs(((ue_azimuth - beam_azimuth + 180) % 360) - 180)
        
        # Elevation alignment
        elevation_diff = abs(ue_elevation - beam_elevation)
        
        # Combined beamwidth (narrower for Sub-THz)
        if bs.freq > 90:
            beamwidth = 10
        else:
            beamwidth = 15
        
        # Calculate gain based on alignment
        if azimuth_diff < beamwidth / 2 and elevation_diff < beamwidth / 2:
            gain = bs.beamforming_gain_db
        elif azimuth_diff < beamwidth and elevation_diff < beamwidth:
            gain = bs.beamforming_gain_db - 3
        else:
            angle_penalty = np.sqrt(azimuth_diff**2 + elevation_diff**2)
            gain = bs.beamforming_gain_db - 20 - 0.15 * (angle_penalty - beamwidth)
            gain = max(gain, 0)
        
        return gain
    
    def _frequencies_interfere(self, freq1: float, freq2: float) -> bool:
        """Check if two frequencies interfere"""
        if abs(freq1 - freq2) < 5:
            return True
        elif abs(freq1 - freq2) < 15:
            return True
        return False
    
    def _calculate_sinr(self, user: UserEquipment6G, bs: BaseStation6G, 
                       current_loads: Dict[int, int] = None) -> float:
        """Calculate SINR with 6G enhancements"""
        d = self._distance_3d(user.x, user.y, user.z, bs.x, bs.y, bs.z)
        
        channel = self._get_or_create_channel(user, bs)
        is_los = channel['los']
        shadow_fade = channel['shadow']
        small_scale_fade = channel['small_scale']
        
        pl = self._path_loss_6g(d, bs.freq, is_los, bs.z, user.z)
        bf_gain = self._calculate_beamforming_gain_3d(bs, user)
        
        rx_power = bs.power + bf_gain - pl + shadow_fade + small_scale_fade
        
        # Interference (reduced for Sub-THz due to narrow beams)
        interference = 0
        for other_bs in self.base_stations:
            if other_bs.id == bs.id:
                continue
            
            other_load = current_loads.get(other_bs.id, 0) if current_loads else other_bs.load
            if other_load == 0:
                continue
            
            if not self._frequencies_interfere(bs.freq, other_bs.freq):
                continue
            
            d_int = self._distance_3d(user.x, user.y, user.z, other_bs.x, other_bs.y, other_bs.z)
            
            max_int_range = self._calculate_coverage_radius(other_bs) * 1.5
            if d_int > max_int_range:
                continue
            
            channel_int = self._get_or_create_channel(user, other_bs)
            is_los_int = channel_int['los']
            shadow_fade_int = channel_int['shadow']
            small_scale_fade_int = channel_int['small_scale']
            
            pl_int = self._path_loss_6g(d_int, other_bs.freq, is_los_int, other_bs.z, user.z)
            
            # Reduced interference for Sub-THz due to directionality
            interference_reduction = 10 if other_bs.freq > 90 else 5
            rx_power_int = other_bs.power - pl_int + shadow_fade_int + small_scale_fade_int - interference_reduction
            interference += 10 ** (rx_power_int / 10)
        
        noise = 10 ** (self.noise_power / 10)
        signal = 10 ** (rx_power / 10)
        
        sinr = 10 * np.log10(signal / (interference + noise))
        sinr = np.clip(sinr, MIN_SINR_DB, MAX_SINR_DB)
        
        return sinr
    
    def _calculate_throughput(self, sinr: float, bandwidth_mhz: float) -> float:
        """Calculate throughput with 6G spectral efficiency"""
        sinr_linear = 10 ** (sinr / 10)
        spectral_efficiency = np.log2(1 + sinr_linear)
        spectral_efficiency = min(spectral_efficiency, MAX_SPECTRAL_EFFICIENCY_6G)
        throughput_mbps = bandwidth_mhz * spectral_efficiency
        return throughput_mbps
    
    def _calculate_reward(self, throughput: float, handover_occurred: bool,
                         prev_throughput: float = 0, sinr: float = 0) -> float:
        """6G reward with latency bonus"""
        reward = throughput / 1000.0  # Normalized
        
        if handover_occurred:
            throughput_improvement = (throughput - prev_throughput) / 1000.0
            if throughput_improvement < 0.025:
                reward -= 0.1
        
        # Low-latency bonus for high SINR
        if sinr > 20:
            reward += 0.05
        
        return reward
    
    def _calculate_coverage_radius(self, bs: BaseStation6G, sinr_threshold: float = 0.0) -> float:
        """Calculate effective coverage radius"""
        min_dist, max_dist = 1, self.grid_size
        target_dist = max_dist
        
        for _ in range(20):
            test_dist = (min_dist + max_dist) / 2
            is_los = self._los_probability_6g(test_dist, bs.freq, bs.bs_type) > 0.5
            pl = self._path_loss_6g(test_dist, bs.freq, is_los, bs.z, 1.5)
            rx_power = bs.power + bs.beamforming_gain_db - pl
            
            avg_interference_power = -100
            noise = 10 ** (self.noise_power / 10)
            interference = 10 ** (avg_interference_power / 10)
            signal = 10 ** (rx_power / 10)
            
            estimated_sinr = 10 * np.log10(signal / (interference + noise))
            
            if estimated_sinr > sinr_threshold:
                min_dist = test_dist
                target_dist = test_dist
            else:
                max_dist = test_dist
        
        return target_dist
    
    def reset(self):
        """Reset simulation"""
        for user in self.users:
            user.x = np.random.uniform(0, self.grid_size)
            user.y = np.random.uniform(0, self.grid_size)
            user.z = 1.5
            user.vx = np.random.uniform(-2, 2)
            user.vy = np.random.uniform(-2, 2)
            user.connected_bs = None
            user.beam_id = None
            user.sinr = 0.0
            user.throughput = 0.0
            user.handover_count = 0
            user.time_since_last_handover = 0
            user.candidate_bs = None
            user.ttt_counter = 0
            user.outage_count = 0
        
        for bs in self.base_stations:
            bs.load = 0
            bs.active_beams.clear()
            bs.beam_angles.clear()
        
        self.ucb.reset()
        self.step = 0
        self.channel_cache = {}
        self._initialize_shadow_map()
        self.step_metrics = {k: [] for k in self.step_metrics}
        self.throughput_distribution = []
        self.sinr_samples = []
    
    def simulation_step(self) -> Dict:
        """Execute one simulation step with TTT handover logic"""
        
        self.channel_cache = {}
        
        # Move users
        for user in self.users:
            user.x = np.clip(user.x + user.vx, 0, self.grid_size)
            user.y = np.clip(user.y + user.vy, 0, self.grid_size)
            
            if user.x <= 0 or user.x >= self.grid_size:
                user.vx = -user.vx
            if user.y <= 0 or user.y >= self.grid_size:
                user.vy = -user.vy
            
            user.time_since_last_handover += 1
        
        prev_throughputs = {u.id: u.throughput for u in self.users}
        
        for bs in self.base_stations:
            bs.load = 0
            bs.active_beams.clear()
        
        current_loads = {bs.id: 0 for bs in self.base_stations}
        
        total_sinr = 0
        total_throughput = 0
        handover_count = 0
        pingpong_count = 0
        total_reward = 0
        connected_users = 0
        outage_count = 0
        throughputs = []
        sinrs = []
        
        OUTAGE_THRESHOLD = 1.0  # 1 Mbps
        
        for user in self.users:
            prev_bs = user.connected_bs
            prev_tput = prev_throughputs[user.id]
            
            # UCB arm selection (no context needed)
            selected_arm, _, should_handover = self.ucb.select_arm(
                ue_id=user.id,
                current_arm=prev_bs,
                handover_penalty=0.2
            )
            
            actual_handover = False
            final_bs_id = prev_bs
            
            if should_handover and prev_bs is not None:
                if user.candidate_bs == selected_arm:
                    user.ttt_counter += 1
                else:
                    user.candidate_bs = selected_arm
                    user.ttt_counter = 1
                
                if user.ttt_counter >= user.ttt_threshold:
                    actual_handover = True
                    final_bs_id = selected_arm
                    user.ttt_counter = 0
                    user.candidate_bs = None
                    
                    # Ping-pong detection
                    if user.time_since_last_handover < 10:
                        pingpong_count += 1
                else:
                    final_bs_id = prev_bs
            else:
                final_bs_id = selected_arm
                user.candidate_bs = None
                user.ttt_counter = 0
            
            selected_bs = self.base_stations[final_bs_id]
            
            if current_loads[selected_bs.id] < selected_bs.max_simultaneous_beams:
                sinr = self._calculate_sinr(user, selected_bs, current_loads)
                throughput = self._calculate_throughput(sinr, selected_bs.bandwidth_mhz)
                
                if actual_handover:
                    handover_count += 1
                    user.handover_count += 1
                    user.time_since_last_handover = 0
                
                reward = self._calculate_reward(throughput, actual_handover, prev_tput, sinr)
                
                # Update UCB with observed reward
                self.ucb.update(user.id, final_bs_id, reward)
                
                beam_id = current_loads[selected_bs.id]
                beam_azimuth = self._angle_between(selected_bs.x, selected_bs.y, user.x, user.y)
                beam_elevation = self._elevation_angle(selected_bs.x, selected_bs.y, selected_bs.z, 
                                                       user.x, user.y, user.z)
                selected_bs.active_beams[user.id] = beam_id
                selected_bs.beam_angles[beam_id] = (beam_azimuth, beam_elevation)
                selected_bs.load += 1
                current_loads[selected_bs.id] += 1
                
                user.connected_bs = selected_bs.id
                user.beam_id = beam_id
                user.sinr = sinr
                user.throughput = throughput
                
                if throughput < OUTAGE_THRESHOLD:
                    outage_count += 1
                    user.outage_count += 1
                
                total_sinr += sinr
                total_throughput += throughput
                total_reward += reward
                connected_users += 1
                throughputs.append(throughput)
                sinrs.append(sinr)
            else:
                # Fallback logic
                alt_options = []
                for i in range(self.num_bs):
                    if current_loads[i] < self.base_stations[i].max_simultaneous_beams:
                        # Estimate quality based on current SINR calculation
                        alt_bs = self.base_stations[i]
                        alt_sinr = self._calculate_sinr(user, alt_bs, current_loads)
                        alt_options.append((i, alt_sinr))
                
                alt_options.sort(key=lambda x: x[1], reverse=True)
                
                assigned = False
                for alt_arm, alt_sinr_est in alt_options:
                    alt_bs = self.base_stations[alt_arm]
                    if current_loads[alt_bs.id] < alt_bs.max_simultaneous_beams:
                        sinr = self._calculate_sinr(user, alt_bs, current_loads)
                        throughput = self._calculate_throughput(sinr, alt_bs.bandwidth_mhz)
                        
                        handover_occurred = (prev_bs is not None and prev_bs != alt_arm)
                        if handover_occurred:
                            handover_count += 1
                            user.handover_count += 1
                            user.time_since_last_handover = 0
                            if user.time_since_last_handover < 10:
                                pingpong_count += 1
                        
                        reward = self._calculate_reward(throughput, handover_occurred, prev_tput, sinr)
                        
                        # Update UCB
                        self.ucb.update(user.id, alt_arm, reward)
                        
                        beam_id = current_loads[alt_bs.id]
                        beam_azimuth = self._angle_between(alt_bs.x, alt_bs.y, user.x, user.y)
                        beam_elevation = self._elevation_angle(alt_bs.x, alt_bs.y, alt_bs.z,
                                                               user.x, user.y, user.z)
                        alt_bs.active_beams[user.id] = beam_id
                        alt_bs.beam_angles[beam_id] = (beam_azimuth, beam_elevation)
                        alt_bs.load += 1
                        current_loads[alt_bs.id] += 1
                        
                        user.connected_bs = alt_arm
                        user.beam_id = beam_id
                        user.sinr = sinr
                        user.throughput = throughput
                        
                        if throughput < OUTAGE_THRESHOLD:
                            outage_count += 1
                            user.outage_count += 1
                        
                        total_sinr += sinr
                        total_throughput += throughput
                        total_reward += reward
                        connected_users += 1
                        throughputs.append(throughput)
                        sinrs.append(sinr)
                        assigned = True
                        break
                
                if not assigned:
                    user.connected_bs = None
                    user.beam_id = None
                    user.sinr = MIN_SINR_DB
                    user.throughput = 0
                    outage_count += 1
                    user.outage_count += 1
        
        self.ucb.total_reward += total_reward
        optimal_reward_estimate = self.num_ues * 0.5
        self.ucb.optimal_reward += optimal_reward_estimate
        cumulative_regret = self.ucb.optimal_reward - self.ucb.total_reward
        
        avg_sinr = total_sinr / max(connected_users, 1)
        coverage = (connected_users / self.num_ues) * 100
        avg_load = np.mean([bs.load for bs in self.base_stations])
        
        # 6G-specific metrics
        if throughputs:
            cell_edge_rate = np.percentile(throughputs, 5)
            self.throughput_distribution.extend(throughputs)
            self.sinr_samples.extend(sinrs)
        else:
            cell_edge_rate = 0
        
        outage_prob = (outage_count / self.num_ues) * 100
        pingpong_rate = (pingpong_count / max(handover_count, 1)) * 100 if handover_count > 0 else 0
        
        metrics = {
            'avg_sinr': avg_sinr,
            'total_throughput': total_throughput,
            'handover_count': handover_count,
            'coverage': coverage,
            'regret': cumulative_regret,
            'avg_load': avg_load,
            'cell_edge_rate': cell_edge_rate,
            'outage_prob': outage_prob,
            'pingpong_rate': pingpong_rate
        }
        
        for key in self.step_metrics:
            if key in metrics:
                self.step_metrics[key].append(metrics[key])
        
        self.step += 1
        
        if self.step % 50 == 0:
            self._add_log(f'SINR: {metrics["avg_sinr"]:.1f}dB, Throughput: {metrics["total_throughput"]:.1f}Mbps, HO: {handover_count}')
        
        return metrics
    
    def run_simulation(self, num_steps: int = 300) -> Dict:
        """Run 6G simulation"""
        self.reset()
        
        print(f"\n{'='*80}")
        print(f"6G GROUND HETNET SIMULATION WITH UCB")
        print(f"{'='*80}")
        print(f"Network: {self.num_macro}M + {self.num_micro}m + {self.num_pico}p")
        print(f"UEs: {self.num_ues} (mobile ground)")
        print(f"Steps: {num_steps}")
        print(f"\n6G ENHANCEMENTS:")
        print(f"  [1] Frequency bands: Sub-6, mmWave, Sub-THz (95-140 GHz)")
        print(f"  [2] Bandwidth: Up to 5 GHz for Pico cells")
        print(f"  [3] Massive antennas: Up to 512 elements")
        print(f"  [4] 3D beamforming with elevation angle")
        print(f"  [5] Sub-THz molecular absorption modeling")
        print(f"  [6] Dense deployment: {self.num_bs} total BSs")
        print(f"  [7] SE cap: {MAX_SPECTRAL_EFFICIENCY_6G} bits/s/Hz")
        print(f"  [8] Beam capacity: Up to 256 simultaneous beams/BS")
        print(f"  [9] Handover logic: TTT=5, penalty=0.2+0.15")
        print(f"  [10] Algorithm: Standard UCB (c={self.ucb_c})")
        print(f"{'='*80}\n")
        
        for step in range(num_steps):
            metrics = self.simulation_step()
            
            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}/{num_steps} | "
                      f"SINR: {metrics['avg_sinr']:.1f}dB | "
                      f"Throughput: {metrics['total_throughput']:.0f}Mbps | "
                      f"Cell-edge: {metrics['cell_edge_rate']:.1f}Mbps | "
                      f"HO: {metrics['handover_count']} | "
                      f"Coverage: {metrics['coverage']:.1f}%")
        
        summary = {
            'total_steps': num_steps,
            'avg_sinr': np.mean(self.step_metrics['avg_sinr']),
            'avg_throughput': np.mean(self.step_metrics['total_throughput']),
            'total_handovers': int(np.sum(self.step_metrics['handover_count'])),
            'avg_coverage': np.mean(self.step_metrics['coverage']),
            'final_regret': self.step_metrics['regret'][-1] if self.step_metrics['regret'] else 0,
            'avg_load': np.mean(self.step_metrics['avg_load']),
            'avg_cell_edge_rate': np.mean(self.step_metrics['cell_edge_rate']),
            'avg_outage_prob': np.mean(self.step_metrics['outage_prob']),
            'avg_pingpong_rate': np.mean(self.step_metrics['pingpong_rate'])
        }
        
        return summary
    
    def print_summary(self):
        """Print simulation summary"""
        print(f"\n{'='*80}")
        print("6G HETNET SIMULATION SUMMARY (UCB)")
        print(f"{'='*80}")
        
        print(f"\n--- 6G ENHANCEMENTS ---")
        print(f"Frequency range: 3.5 - 140 GHz")
        print(f"Max bandwidth: 5 GHz (Pico cells)")
        print(f"Max antennas: 512 elements")
        print(f"SE cap: {MAX_SPECTRAL_EFFICIENCY_6G} bits/s/Hz")
        print(f"3D beamforming: Azimuth + Elevation")
        print(f"Algorithm: Standard UCB (c={self.ucb_c})")
        
        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"Avg SINR: {np.mean(self.step_metrics['avg_sinr']):.2f} dB")
        print(f"Avg Total Throughput: {np.mean(self.step_metrics['total_throughput']):.1f} Mbps")
        print(f"Avg Cell-Edge Rate: {np.mean(self.step_metrics['cell_edge_rate']):.1f} Mbps (5th percentile)")
        print(f"Total Handovers: {int(np.sum(self.step_metrics['handover_count']))}")
        print(f"Avg Ping-Pong Rate: {np.mean(self.step_metrics['pingpong_rate']):.1f}%")
        print(f"Avg Coverage: {np.mean(self.step_metrics['coverage']):.1f}%")
        print(f"Avg Outage Probability: {np.mean(self.step_metrics['outage_prob']):.2f}%")
        
        if self.throughput_distribution:
            print(f"\n--- THROUGHPUT DISTRIBUTION ---")
            print(f"Median: {np.median(self.throughput_distribution):.1f} Mbps")
            print(f"95th percentile: {np.percentile(self.throughput_distribution, 95):.1f} Mbps")
            print(f"Max: {np.max(self.throughput_distribution):.1f} Mbps")
        
        if self.sinr_samples:
            print(f"\n--- SINR DISTRIBUTION ---")
            print(f"Median: {np.median(self.sinr_samples):.1f} dB")
            print(f"5th percentile: {np.percentile(self.sinr_samples, 5):.1f} dB")
        
        print(f"\n{'='*80}\n")


# Main execution
if __name__ == "__main__":
    sim = HetNet6GSimulator(
        grid_size=3000,
        num_macro=4,
        num_micro=8,
        num_pico=12,
        num_ues=300,
        tx_power_macro=43,
        noise_power=-174,
        ucb_c=2.0  # UCB exploration parameter
    )
    
    summary = sim.run_simulation(num_steps=100)
    sim.print_summary()