import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from collections import defaultdict
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# Custom Gaussian smoothing filter (scipy-free alternative)
def gaussian_filter(array, sigma=3):
    """Simple Gaussian smoothing using numpy convolution (scipy-free)"""
    size = int(sigma * 4)
    x = np.arange(-size, size + 1)
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    
    # Apply separable 2D convolution  
    result = array.copy()
    for i in range(result.shape[0]):
        result[i, :] = np.convolve(result[i, :], kernel_1d, mode='same')
    for j in range(result.shape[1]):
        result[:, j] = np.convolve(result[:, j], kernel_1d, mode='same')
    
    return result

# Frequency band to bandwidth mapping (5G NR)
BAND_BANDWIDTH = {
    24: 100,   # n258: 100 MHz
    28: 100,   # n257: 100 MHz
    47: 200,   # n259: 200 MHz
    60: 400,   # Unlicensed: 400 MHz
    73: 400,   # FR2-2: 400 MHz
    100: 400,  # Future bands: 400 MHz
}

# CRITICAL FIX 1: Realistic spectral efficiency cap (3GPP limits)
MAX_SPECTRAL_EFFICIENCY = 7.8  # bits/s/Hz (realistic for 5G NR with 256-QAM)

# CRITICAL FIX 2: SINR bounds to prevent numerical issues
MIN_SINR_DB = -10.0
MAX_SINR_DB = 30.0  # Realistic upper bound for practical systems

@dataclass
class BaseStation:
    id: int
    x: float
    y: float
    freq: float
    power: float
    beam_direction: float
    bs_type: str  # 'macro', 'micro', 'pico', 'femto'
    
    # Massive MIMO parameters
    num_antennas: int = 64
    antenna_spacing: float = 0.5  # in wavelengths
    max_simultaneous_beams: int = 8
    beamforming_gain_db: float = 20.0
    
    # Bandwidth
    bandwidth_mhz: float = 100.0
    
    # State
    load: int = 0
    active_beams: Dict[int, int] = field(default_factory=dict)  # {ue_id: beam_id}
    beam_angles: Dict[int, float] = field(default_factory=dict)  # {beam_id: angle}

@dataclass
class UserEquipment:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    
    # UE antenna configuration
    num_antennas: int = 4
    
    # Connection state
    connected_bs: Optional[int] = None
    beam_id: Optional[int] = None
    sinr: float = 0.0
    throughput: float = 0.0
    
    # CRITICAL FIX 3: Time-to-Trigger (TTT) for handover hysteresis
    candidate_bs: Optional[int] = None
    ttt_counter: int = 0
    ttt_threshold: int = 5  # Need 5 consecutive steps before handover
    
    # Handover tracking
    handover_count: int = 0
    time_since_last_handover: int = 0

class ContextualLinUCB:
    """Contextual Multi-Armed Bandit using LinUCB for handover decisions"""
    
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Separate model for each UE
        self.A = {}  # {ue_id: [A_matrix per arm]}
        self.b = {}  # {ue_id: [b_vector per arm]}
        
        self.total_reward = 0.0
        self.optimal_reward = 0.0
        self.decision_count = 0
    
    def _initialize_ue(self, ue_id: int):
        """Initialize LinUCB parameters for a new UE"""
        if ue_id not in self.A:
            self.A[ue_id] = [np.identity(self.n_features) for _ in range(self.n_arms)]
            self.b[ue_id] = [np.zeros(self.n_features) for _ in range(self.n_arms)]
    
    def select_arm(self, ue_id: int, contexts: List[np.ndarray], 
                   current_arm: Optional[int] = None,
                   handover_penalty: float = 0.2) -> Tuple[int, List[Dict], bool]:
        """
        FIXED: Stronger handover penalty with hysteresis margin
        """
        self._initialize_ue(ue_id)
        
        ucb_values = []
        best_arm = 0
        best_ucb = -np.inf
        
        for arm in range(self.n_arms):
            context = contexts[arm]
            try:
                A_inv = np.linalg.inv(self.A[ue_id][arm])
                theta = A_inv @ self.b[ue_id][arm]
                expected_reward = theta @ context
                
                # FIXED: Apply stronger handover penalty AND hysteresis margin
                if current_arm is not None and arm != current_arm:
                    expected_reward -= handover_penalty
                    # Additional hysteresis: new BS must be significantly better
                    hysteresis_margin = 0.15
                    expected_reward -= hysteresis_margin
                
                # Calculate UCB with penalized expected reward
                uncertainty = self.alpha * np.sqrt(context @ A_inv @ context)
                ucb = expected_reward + uncertainty
                
                ucb_values.append({
                    'arm': arm, 
                    'ucb': ucb, 
                    'expected_reward': expected_reward, 
                    'uncertainty': uncertainty
                })
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_arm = arm
                    
            except np.linalg.LinAlgError:
                ucb_values.append({'arm': arm, 'ucb': 0, 'expected_reward': 0, 'uncertainty': 0})
        
        should_handover = (current_arm is None) or (best_arm != current_arm)
        return best_arm, ucb_values, should_handover
    
    def update(self, ue_id: int, arm: int, context: np.ndarray, reward: float):
        """Update LinUCB model for specific UE-BS pair"""
        self._initialize_ue(ue_id)
        self.A[ue_id][arm] += np.outer(context, context)
        self.b[ue_id][arm] += reward * context
        self.decision_count += 1
    
    def reset(self):
        """Reset all learning parameters"""
        self.A = {}
        self.b = {}
        self.total_reward = 0.0
        self.optimal_reward = 0.0
        self.decision_count = 0


class HetNet5GSimulator:
    """5G HetNet simulator - COMPREHENSIVELY FIXED VERSION"""
    
    def __init__(self, grid_size: int = 3000, 
                 num_macro: int = 3, num_micro: int = 4, num_pico: int = 6,
                 num_ues: int = 1000, tx_power_macro: float = 43, 
                 noise_power: float = -174, linucb_alpha: float = 1.0):
        
        self.grid_size = grid_size
        self.num_macro = num_macro
        self.num_micro = num_micro
        self.num_pico = num_pico
        self.num_bs = num_macro + num_micro + num_pico
        self.num_ues = num_ues
        self.tx_power_macro = tx_power_macro
        self.noise_power = noise_power
        self.linucb_alpha = linucb_alpha
        
        # Network components
        self.base_stations: List[BaseStation] = []
        self.users: List[UserEquipment] = []
        self.cmab: ContextualLinUCB = None
        
        # Channel state cache per time step
        self.channel_cache = {}
        self.shadow_fading_map = {}
        
        # Tracking
        self.step = 0
        self.logs = []
        self.step_metrics = {
            'avg_sinr': [], 'total_throughput': [], 'handover_count': [],
            'coverage': [], 'regret': [], 'avg_load': []
        }
        
        self._initialize_hetnet()
        self._initialize_shadow_map()
    
    def _initialize_hetnet(self):
        """Initialize HetNet with macro, micro, and pico cells"""
        self.base_stations = self._deploy_hetnet()
        self.users = self._generate_users()
        self.cmab = ContextualLinUCB(
            n_arms=self.num_bs, 
            n_features=8,
            alpha=self.linucb_alpha
        )
        self._add_log(f'HetNet initialized: {self.num_macro}M + {self.num_micro}m + {self.num_pico}p BSs, {self.num_ues} UEs')
    
    def _initialize_shadow_map(self):
        """Initialize spatially correlated shadow fading map"""
        grid_points = 50
        x = np.linspace(0, self.grid_size, grid_points)
        y = np.linspace(0, self.grid_size, grid_points)
        
        for bs in self.base_stations:
            shadow_map = np.random.normal(0, 4.0, (grid_points, grid_points))
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
        
        return np.clip(shadow_map['values'][y_idx, x_idx], -12, 12)
    
    def _deploy_hetnet(self) -> List[BaseStation]:
        """Deploy heterogeneous network with different BS types"""
        bs_list = []
        bs_id = 0
        
        # 1. MACRO CELLS
        for i in range(self.num_macro):
            angle = (2 * np.pi * i) / self.num_macro
            radius = self.grid_size / 3
            x = self.grid_size / 2 + radius * np.cos(angle)
            y = self.grid_size / 2 + radius * np.sin(angle)
            
            freq = np.random.choice([24, 28])
            bs_list.append(BaseStation(
                id=bs_id, x=x, y=y,
                freq=freq,
                power=self.tx_power_macro,
                beam_direction=np.random.uniform(0, 360),
                bs_type='macro',
                num_antennas=128,
                max_simultaneous_beams=32,
                beamforming_gain_db=23.0,
                bandwidth_mhz=BAND_BANDWIDTH.get(int(freq), 100)
            ))
            bs_id += 1
        
        # 2. MICRO CELLS
        for i in range(self.num_micro):
            x = np.random.uniform(self.grid_size * 0.2, self.grid_size * 0.8)
            y = np.random.uniform(self.grid_size * 0.2, self.grid_size * 0.8)
            
            freq = np.random.choice([28, 39, 47])
            bs_list.append(BaseStation(
                id=bs_id, x=x, y=y,
                freq=freq,
                power=self.tx_power_macro - 10,
                beam_direction=np.random.uniform(0, 360),
                bs_type='micro',
                num_antennas=64,
                max_simultaneous_beams=16,
                beamforming_gain_db=20.0,
                bandwidth_mhz=BAND_BANDWIDTH.get(int(freq), 100)
            ))
            bs_id += 1
        
        # 3. PICO CELLS
        for i in range(self.num_pico):
            x = np.random.uniform(0, self.grid_size)
            y = np.random.uniform(0, self.grid_size)
            
            freq = np.random.choice([39, 47, 60, 73])
            bs_list.append(BaseStation(
                id=bs_id, x=x, y=y,
                freq=freq,
                power=self.tx_power_macro - 20,
                beam_direction=np.random.uniform(0, 360),
                bs_type='pico',
                num_antennas=32,
                max_simultaneous_beams=8,
                beamforming_gain_db=17.0,
                bandwidth_mhz=BAND_BANDWIDTH.get(int(freq), 200)
            ))
            bs_id += 1
        
        return bs_list
    
    def _generate_users(self) -> List[UserEquipment]:
        """Generate mobile users with random velocities"""
        return [UserEquipment(
            id=i,
            x=np.random.uniform(0, self.grid_size),
            y=np.random.uniform(0, self.grid_size),
            vx=np.random.uniform(-2, 2),
            vy=np.random.uniform(-2, 2),
            num_antennas=np.random.choice([2, 4, 8])
        ) for i in range(self.num_ues)]
    
    def _add_log(self, message: str):
        self.logs.append(f'[S{self.step}] {message}')
        if len(self.logs) > 30:
            self.logs = self.logs[-30:]
    
    @staticmethod
    def _distance(x1, y1, x2, y2) -> float:
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    @staticmethod
    def _angle_between(x1, y1, x2, y2) -> float:
        """Calculate angle in degrees"""
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    
    @staticmethod
    def _los_probability(d: float, bs_type: str) -> float:
        """LOS probability based on distance and BS type"""
        if bs_type == 'macro':
            d0, d1 = 250, 150
        elif bs_type == 'micro':
            d0, d1 = 150, 80
        else:
            d0, d1 = 80, 40
        return min(d0 / max(d, 1), 1) * (1 - np.exp(-d / d1)) + np.exp(-d / d1)
    
    @staticmethod
    def _path_loss(d: float, fc: float, is_los: bool) -> float:
        """3GPP TR 38.901 path loss model for 5G mmWave"""
        d = max(d, 1)
        
        if is_los:
            pl = 32.4 + 20 * np.log10(d) + 20 * np.log10(fc)
        else:
            pl_los = 32.4 + 20 * np.log10(d) + 20 * np.log10(fc)
            nlos_penalty = 20 + 0.4 * (fc - 24)
            pl = pl_los + nlos_penalty
        
        return pl
    
    @staticmethod
    def _rayleigh_fading() -> float:
        """Generate Rayleigh fading (NLOS channels)"""
        real = np.random.normal(0, 1)
        imag = np.random.normal(0, 1)
        magnitude = np.sqrt(real**2 + imag**2)
        fading_db = 20 * np.log10(magnitude / np.sqrt(2))
        return np.clip(fading_db, -10, 6)
    
    @staticmethod
    def _rician_fading(k_factor_db: float = 10.0) -> float:
        """Generate Rician fading (LOS channels)"""
        k_linear = 10 ** (k_factor_db / 10)
        
        los_component = np.sqrt(k_linear / (k_linear + 1))
        
        scatter_power = 1 / (k_linear + 1)
        real = np.random.normal(0, np.sqrt(scatter_power / 2))
        imag = np.random.normal(0, np.sqrt(scatter_power / 2))
        
        total_real = los_component + real
        total_imag = imag
        magnitude = np.sqrt(total_real**2 + total_imag**2)
        
        fading_db = 20 * np.log10(magnitude)
        return np.clip(fading_db, -5, 8)
    
    def _get_or_create_channel(self, user: UserEquipment, bs: BaseStation) -> Dict:
        """Get or create channel state with coherence"""
        key = (user.id, bs.id)
        
        if key not in self.channel_cache:
            d = self._distance(user.x, user.y, bs.x, bs.y)
            
            # LOS state (changes slowly)
            los_seed = (self.step // 10) * 1000 + user.id * 100 + bs.id
            np.random.seed(los_seed)
            is_los = np.random.uniform(0, 1) < self._los_probability(d, bs.bs_type)
            
            # Shadow fading (spatially correlated)
            shadow_fade = self._get_shadow_fading(bs.id, user.x, user.y)
            
            # Small-scale fading (changes every 5 steps)
            fade_seed = (self.step // 5) * 1000 + user.id * 100 + bs.id
            np.random.seed(fade_seed)
            if is_los:
                k_factor = max(5, 15 - d/100)
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
    
    def _calculate_beamforming_gain(self, bs: BaseStation, user: UserEquipment) -> float:
        """Calculate beamforming gain based on angle alignment"""
        ue_angle = self._angle_between(bs.x, bs.y, user.x, user.y)
        
        if user.id in bs.active_beams:
            beam_angle = bs.beam_angles[bs.active_beams[user.id]]
        else:
            beam_angle = ue_angle
        
        angle_diff = abs(((ue_angle - beam_angle + 180) % 360) - 180)
        
        beamwidth = 15
        
        if angle_diff < beamwidth / 2:
            gain = bs.beamforming_gain_db
        elif angle_diff < beamwidth:
            gain = bs.beamforming_gain_db - 3
        else:
            gain = bs.beamforming_gain_db - 15 - 0.1 * (angle_diff - beamwidth)
            gain = max(gain, 0)
        
        return gain
    
    def _frequencies_interfere(self, freq1: float, freq2: float) -> bool:
        """Check if two frequencies interfere"""
        if abs(freq1 - freq2) < 5:
            return True
        if abs(freq1 - freq2) < 15:
            return True
        return False
    
    def _calculate_sinr(self, user: UserEquipment, bs: BaseStation, 
                       current_loads: Dict[int, int] = None) -> float:
        """FIXED: Calculate SINR with bounds checking"""
        d = self._distance(user.x, user.y, bs.x, bs.y)
        
        channel = self._get_or_create_channel(user, bs)
        is_los = channel['los']
        shadow_fade = channel['shadow']
        small_scale_fade = channel['small_scale']
        
        pl = self._path_loss(d, bs.freq, is_los)
        bf_gain = self._calculate_beamforming_gain(bs, user)
        
        rx_power = bs.power + bf_gain - pl + shadow_fade + small_scale_fade
        
        # Frequency-aware interference
        interference = 0
        for other_bs in self.base_stations:
            if other_bs.id == bs.id:
                continue
            
            other_load = current_loads.get(other_bs.id, 0) if current_loads else other_bs.load
            if other_load == 0:
                continue
            
            if not self._frequencies_interfere(bs.freq, other_bs.freq):
                continue
            
            d_int = self._distance(user.x, user.y, other_bs.x, other_bs.y)
            
            max_int_range = self._calculate_coverage_radius(other_bs) * 2
            if d_int > max_int_range:
                continue
            
            channel_int = self._get_or_create_channel(user, other_bs)
            is_los_int = channel_int['los']
            shadow_fade_int = channel_int['shadow']
            small_scale_fade_int = channel_int['small_scale']
            
            pl_int = self._path_loss(d_int, other_bs.freq, is_los_int)
            
            rx_power_int = other_bs.power - pl_int + shadow_fade_int + small_scale_fade_int - 10
            interference += 10 ** (rx_power_int / 10)
        
        noise = 10 ** (self.noise_power / 10)
        signal = 10 ** (rx_power / 10)
        
        sinr = 10 * np.log10(signal / (interference + noise))
        
        # CRITICAL FIX 4: Bound SINR to realistic range
        sinr = np.clip(sinr, MIN_SINR_DB, MAX_SINR_DB)
        
        return sinr
    
    def _calculate_throughput(self, sinr: float, bandwidth_mhz: float) -> float:
        """CRITICAL FIX 5: Properly cap spectral efficiency"""
        sinr_linear = 10 ** (sinr / 10)
        spectral_efficiency = np.log2(1 + sinr_linear)
        
        # Apply realistic cap
        spectral_efficiency = min(spectral_efficiency, MAX_SPECTRAL_EFFICIENCY)
        
        # CRITICAL FIX 6: Correct units - bandwidth in MHz
        throughput_mbps = bandwidth_mhz * spectral_efficiency
        
        return throughput_mbps
    
    def _get_context(self, user: UserEquipment, bs: BaseStation,
                    current_loads: Dict[int, int] = None) -> np.ndarray:
        """Enhanced context vector for CMAB"""
        d = self._distance(user.x, user.y, bs.x, bs.y)
        angle = self._angle_between(bs.x, bs.y, user.x, user.y)
        sinr = self._calculate_sinr(user, bs, current_loads)
        
        current_load = current_loads.get(bs.id, 0) if current_loads else bs.load
        
        # Normalize with realistic bounds
        context = np.array([
            (sinr - MIN_SINR_DB) / (MAX_SINR_DB - MIN_SINR_DB),  # [0, 1]
            d / self.grid_size,
            (angle + 180) / 360,
            bs.freq / 100,
            bs.power / 50,
            bs.bandwidth_mhz / 400,
            bs.num_antennas / 128,
            current_load / max(self.num_ues / self.num_bs * 2, 1)
        ])
        
        return context
    
    def _calculate_reward(self, throughput: float, handover_occurred: bool,
                         prev_throughput: float = 0) -> float:
        """CRITICAL FIX 7: Realistic reward scaling"""
        # Normalize throughput to 0-1 range (assuming max ~1000 Mbps per UE)
        reward = throughput / 1000.0
        
        if handover_occurred:
            throughput_improvement = (throughput - prev_throughput) / 1000.0
            # Stronger penalty if handover doesn't improve by at least 50 Mbps
            if throughput_improvement < 0.05:
                reward -= 0.1
        
        return reward
    
    def _calculate_coverage_radius(self, bs: BaseStation, sinr_threshold: float = 0.0) -> float:
        """Calculate effective coverage radius"""
        min_dist, max_dist = 1, self.grid_size
        target_dist = max_dist
        
        for _ in range(20):
            test_dist = (min_dist + max_dist) / 2
            is_los = self._los_probability(test_dist, bs.bs_type) > 0.5
            pl = self._path_loss(test_dist, bs.freq, is_los)
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
        
        for bs in self.base_stations:
            bs.load = 0
            bs.active_beams.clear()
            bs.beam_angles.clear()
        
        self.cmab.reset()
        self.step = 0
        self.channel_cache = {}
        self._initialize_shadow_map()
        self.step_metrics = {k: [] for k in self.step_metrics}
    
    def simulation_step(self) -> Dict:
        """CRITICAL FIX 8: Execute step with TTT handover logic"""
        
        # Clear channel cache for new step
        self.channel_cache = {}
        
        # 1. Move users
        for user in self.users:
            user.x = np.clip(user.x + user.vx, 0, self.grid_size)
            user.y = np.clip(user.y + user.vy, 0, self.grid_size)
            
            if user.x <= 0 or user.x >= self.grid_size:
                user.vx = -user.vx
            if user.y <= 0 or user.y >= self.grid_size:
                user.vy = -user.vy
            
            user.time_since_last_handover += 1
        
        # 2. Store previous state
        prev_throughputs = {u.id: u.throughput for u in self.users}
        
        # 3. Reset BS loads
        for bs in self.base_stations:
            bs.load = 0
            bs.active_beams.clear()
        
        # 4. Track current loads
        current_loads = {bs.id: 0 for bs in self.base_stations}
        
        # 5. CMAB-based BS selection WITH TTT
        total_sinr = 0
        total_throughput = 0
        handover_count = 0
        total_reward = 0
        connected_users = 0
        
        for user in self.users:
            prev_bs = user.connected_bs
            prev_tput = prev_throughputs[user.id]
            
            # Get contexts
            contexts = [self._get_context(user, bs, current_loads) for bs in self.base_stations]
            
            # Select best arm
            selected_arm, _, should_handover = self.cmab.select_arm(
                ue_id=user.id,
                contexts=contexts,
                current_arm=prev_bs,
                handover_penalty=0.2  # Stronger penalty
            )
            
            # CRITICAL FIX 9: Time-to-Trigger (TTT) logic
            actual_handover = False
            final_bs_id = prev_bs
            
            if should_handover and prev_bs is not None:
                # Check if this is the same candidate
                if user.candidate_bs == selected_arm:
                    user.ttt_counter += 1
                else:
                    # New candidate - reset counter
                    user.candidate_bs = selected_arm
                    user.ttt_counter = 1
                
                # Only handover if TTT threshold is met
                if user.ttt_counter >= user.ttt_threshold:
                    actual_handover = True
                    final_bs_id = selected_arm
                    user.ttt_counter = 0
                    user.candidate_bs = None
                else:
                    # Stay with current BS during TTT period
                    final_bs_id = prev_bs
            else:
                # No handover needed or initial connection
                final_bs_id = selected_arm
                user.candidate_bs = None
                user.ttt_counter = 0
            
            selected_bs = self.base_stations[final_bs_id]
            
            # Check capacity
            if current_loads[selected_bs.id] < selected_bs.max_simultaneous_beams:
                sinr = self._calculate_sinr(user, selected_bs, current_loads)
                throughput = self._calculate_throughput(sinr, selected_bs.bandwidth_mhz)
                
                if actual_handover:
                    handover_count += 1
                    user.handover_count += 1
                    user.time_since_last_handover = 0
                
                reward = self._calculate_reward(throughput, actual_handover, prev_tput)
                
                self.cmab.update(user.id, final_bs_id, contexts[final_bs_id], reward)
                
                beam_id = current_loads[selected_bs.id]
                beam_angle = self._angle_between(selected_bs.x, selected_bs.y, user.x, user.y)
                selected_bs.active_beams[user.id] = beam_id
                selected_bs.beam_angles[beam_id] = beam_angle
                selected_bs.load += 1
                current_loads[selected_bs.id] += 1
                
                user.connected_bs = selected_bs.id
                user.beam_id = beam_id
                user.sinr = sinr
                user.throughput = throughput
                
                total_sinr += sinr
                total_throughput += throughput
                total_reward += reward
                connected_users += 1
            else:
                # Fallback: find available BS
                alt_options = []
                for i in range(self.num_bs):
                    if current_loads[i] < self.base_stations[i].max_simultaneous_beams:
                        alt_context = self._get_context(user, self.base_stations[i], current_loads)
                        alt_sinr_estimate = alt_context[0] * (MAX_SINR_DB - MIN_SINR_DB) + MIN_SINR_DB
                        alt_options.append((i, alt_context, alt_sinr_estimate))
                
                alt_options.sort(key=lambda x: x[2], reverse=True)
                
                assigned = False
                for alt_arm, alt_context, _ in alt_options:
                    alt_bs = self.base_stations[alt_arm]
                    if current_loads[alt_bs.id] < alt_bs.max_simultaneous_beams:
                        sinr = self._calculate_sinr(user, alt_bs, current_loads)
                        throughput = self._calculate_throughput(sinr, alt_bs.bandwidth_mhz)
                        
                        handover_occurred = (prev_bs is not None and prev_bs != alt_arm)
                        if handover_occurred:
                            handover_count += 1
                            user.handover_count += 1
                            user.time_since_last_handover = 0
                        
                        reward = self._calculate_reward(throughput, handover_occurred, prev_tput)
                        
                        self.cmab.update(user.id, alt_arm, alt_context, reward)
                        
                        beam_id = current_loads[alt_bs.id]
                        beam_angle = self._angle_between(alt_bs.x, alt_bs.y, user.x, user.y)
                        alt_bs.active_beams[user.id] = beam_id
                        alt_bs.beam_angles[beam_id] = beam_angle
                        alt_bs.load += 1
                        current_loads[alt_bs.id] += 1
                        
                        user.connected_bs = alt_arm
                        user.beam_id = beam_id
                        user.sinr = sinr
                        user.throughput = throughput
                        
                        total_sinr += sinr
                        total_throughput += throughput
                        total_reward += reward
                        connected_users += 1
                        assigned = True
                        break
                
                if not assigned:
                    user.connected_bs = None
                    user.beam_id = None
                    user.sinr = MIN_SINR_DB
                    user.throughput = 0
        
        # 6. Calculate metrics
        self.cmab.total_reward += total_reward
        optimal_reward_estimate = self.num_ues * 0.5  # More realistic optimal
        self.cmab.optimal_reward += optimal_reward_estimate
        cumulative_regret = self.cmab.optimal_reward - self.cmab.total_reward
        
        avg_sinr = total_sinr / max(connected_users, 1)
        coverage = (connected_users / self.num_ues) * 100
        avg_load = np.mean([bs.load for bs in self.base_stations])
        
        metrics = {
            'avg_sinr': avg_sinr,
            'total_throughput': total_throughput,
            'handover_count': handover_count,
            'coverage': coverage,
            'regret': cumulative_regret,
            'avg_load': avg_load
        }
        
        for key in self.step_metrics:
            if key in metrics:
                self.step_metrics[key].append(metrics[key])
        
        self.step += 1
        
        if self.step % 50 == 0:
            self._add_log(f'SINR: {metrics["avg_sinr"]:.1f}dB, Throughput: {metrics["total_throughput"]:.1f}Mbps, Handovers: {handover_count}')
        
        return metrics
    
    def run_simulation(self, num_steps: int = 300) -> Dict:
        """Run simulation for specified number of steps"""
        self.reset()
        
        print(f"\n{'='*70}")
        print(f"5G HetNet Simulation - COMPREHENSIVELY FIXED")
        print(f"{'='*70}")
        print(f"Network: {self.num_macro} Macro + {self.num_micro} Micro + {self.num_pico} Pico")
        print(f"UEs: {self.num_ues} (mobile)")
        print(f"Steps: {num_steps}")
        print(f"\nCRITICAL FIXES:")
        print(f"  [1] Realistic SE cap: {MAX_SPECTRAL_EFFICIENCY} bits/s/Hz")
        print(f"  [2] SINR bounds: {MIN_SINR_DB} to {MAX_SINR_DB} dB")
        print(f"  [3] Time-to-Trigger (TTT): 5 steps before handover")
        print(f"  [4] Stronger handover penalty: 0.2 + 0.15 hysteresis")
        print(f"  [5] Correct throughput units (Mbps)")
        print(f"  [6] Realistic reward scaling")
        print(f"  [7] Channel coherence + frequency-aware interference")
        print(f"{'='*70}\n")
        
        for step in range(num_steps):
            metrics = self.simulation_step()
            
            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}/{num_steps} | "
                      f"SINR: {metrics['avg_sinr']:.1f}dB | "
                      f"Throughput: {metrics['total_throughput']:.0f}Mbps | "
                      f"Handovers: {metrics['handover_count']} | "
                      f"Coverage: {metrics['coverage']:.1f}%")
        
        summary = {
            'total_steps': num_steps,
            'avg_sinr': np.mean(self.step_metrics['avg_sinr']),
            'avg_throughput': np.mean(self.step_metrics['total_throughput']),
            'total_handovers': int(np.sum(self.step_metrics['handover_count'])),
            'avg_coverage': np.mean(self.step_metrics['coverage']),
            'final_regret': self.step_metrics['regret'][-1] if self.step_metrics['regret'] else 0,
            'avg_load': np.mean(self.step_metrics['avg_load'])
        }
        
        return summary
    
    def _plot_enhanced_topology(self, ax):
        """Enhanced topology visualization"""
        GRID_SIZE = self.grid_size
        ax.set_xlim(-20, GRID_SIZE + 20)
        ax.set_ylim(-20, GRID_SIZE + 20)
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('#f8f9fa')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        
        ax.set_xlabel('X Position (meters)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position (meters)', fontsize=11, fontweight='bold')
        ax.set_title('5G HetNet - COMPREHENSIVELY FIXED', 
                     fontsize=13, fontweight='bold', pad=15)
        
        boundary = Rectangle((0, 0), GRID_SIZE, GRID_SIZE, 
                             fill=False, edgecolor='black', 
                             linewidth=3, linestyle='-')
        ax.add_patch(boundary)
        
        type_colors = {
            'macro': '#e74c3c',
            'micro': '#f39c12',
            'pico': '#27ae60'
        }
        
        type_alphas = {
            'macro': 0.15,
            'micro': 0.12,
            'pico': 0.10
        }
        
        # Coverage circles
        for bs in self.base_stations:
            coverage_radius = self._calculate_coverage_radius(bs, sinr_threshold=0.0)
            circle = Circle((bs.x, bs.y), coverage_radius, 
                           fill=True, 
                           facecolor=type_colors[bs.bs_type],
                           edgecolor=type_colors[bs.bs_type],
                           alpha=type_alphas[bs.bs_type],
                           linewidth=2,
                           linestyle='--',
                           zorder=1)
            ax.add_patch(circle)
        
        # Connections
        for bs in self.base_stations:
            color = type_colors[bs.bs_type]
            for ue_id, beam_id in bs.active_beams.items():
                if ue_id < len(self.users):
                    user = self.users[ue_id]
                    ax.plot([bs.x, user.x], [bs.y, user.y], 
                           color=color, alpha=0.6, linewidth=2, 
                           linestyle='-', zorder=2)
        
        # Base stations
        for bs in self.base_stations:
            color = type_colors[bs.bs_type]
            
            if bs.bs_type == 'macro':
                marker_size = 300
                marker = '^'
            elif bs.bs_type == 'micro':
                marker_size = 200
                marker = 's'
            else:
                marker_size = 150
                marker = 'D'
            
            ax.scatter(bs.x, bs.y, s=marker_size, c=color, marker=marker,
                      edgecolors='white', linewidths=3, zorder=4, alpha=0.9)
            ax.scatter(bs.x, bs.y, s=marker_size, c=color, marker=marker,
                      edgecolors='black', linewidths=1.5, zorder=5, alpha=0.9)
            
            coverage_radius = self._calculate_coverage_radius(bs, sinr_threshold=0.0)
            label_text = (f'{bs.bs_type[0].upper()}{bs.id}\n'
                         f'{bs.freq:.0f}GHz\n'
                         f'{bs.bandwidth_mhz:.0f}MHz\n'
                         f'{bs.load}/{bs.max_simultaneous_beams}')
            
            label_offset = coverage_radius + 25
            ax.text(bs.x, bs.y - label_offset, label_text,
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor=color, linewidth=2, alpha=0.95),
                   zorder=6)
        
        # Users
        for user in self.users:
            if user.connected_bs is not None:
                bs = self.base_stations[user.connected_bs]
                ue_color = type_colors[bs.bs_type]
                edge_color = 'white'
            else:
                ue_color = 'gray'
                edge_color = 'black'
            
            ax.scatter(user.x, user.y, s=80, c=ue_color, marker='o',
                      edgecolors=edge_color, linewidths=2, zorder=7, alpha=0.9)
            
            if abs(user.vx) > 0.1 or abs(user.vy) > 0.1:
                arrow = FancyArrowPatch((user.x, user.y),
                                       (user.x + user.vx * 15, user.y + user.vy * 15),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=2, color=ue_color,
                                       alpha=0.7, zorder=6)
                ax.add_patch(arrow)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=type_colors['macro'], edgecolor='black',
                          linewidth=1.5, label=f'Macro BS (n={self.num_macro})'),
            mpatches.Patch(facecolor=type_colors['micro'], edgecolor='black',
                          linewidth=1.5, label=f'Micro BS (n={self.num_micro})'),
            mpatches.Patch(facecolor=type_colors['pico'], edgecolor='black',
                          linewidth=1.5, label=f'Pico BS (n={self.num_pico})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                      markersize=10, markeredgecolor='white', markeredgewidth=2,
                      label=f'UE (n={self.num_ues})'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                 framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
        
        # Scale bar
        scale_length = 100
        scale_x = GRID_SIZE - 120
        scale_y = 20
        ax.plot([scale_x, scale_x + scale_length], [scale_y, scale_y],
               'k-', linewidth=3, zorder=10)
        ax.plot([scale_x, scale_x], [scale_y - 5, scale_y + 5],
               'k-', linewidth=3, zorder=10)
        ax.plot([scale_x + scale_length, scale_x + scale_length], 
               [scale_y - 5, scale_y + 5], 'k-', linewidth=3, zorder=10)
        ax.text(scale_x + scale_length/2, scale_y - 15, '100m',
               ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='black', alpha=0.9))
        
        # Stats
        if len(self.step_metrics['avg_sinr']) > 0:
            stats_text = (f'SINR: {self.step_metrics["avg_sinr"][-1]:.1f} dB\n'
                         f'Throughput: {self.step_metrics["total_throughput"][-1]:.0f} Mbps\n'
                         f'Coverage: {self.step_metrics["coverage"][-1]:.1f}%\n'
                         f'TTT: 5 steps')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                            edgecolor='black', linewidth=2, alpha=0.9))
    
    def visualize_results(self, save_path: str = None):
        """Comprehensive visualization"""
        fig = plt.figure(figsize=(24, 18))
        
        # Topology
        ax1 = fig.add_subplot(3, 4, (1, 5))
        self._plot_enhanced_topology(ax1)
        
        # SINR
        ax2 = fig.add_subplot(3, 4, 2)
        steps = range(len(self.step_metrics['avg_sinr']))
        ax2.plot(steps, self.step_metrics['avg_sinr'], 'b-', linewidth=2.5)
        ax2.fill_between(steps, self.step_metrics['avg_sinr'], alpha=0.3, color='blue')
        ax2.axhline(y=MAX_SINR_DB, color='red', linestyle='--', label=f'Cap: {MAX_SINR_DB}dB')
        ax2.set_xlabel('Step', fontweight='bold')
        ax2.set_ylabel('SINR (dB)', fontweight='bold')
        ax2.set_title('SINR (Bounded)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        # Throughput
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(steps, self.step_metrics['total_throughput'], 'g-', linewidth=2.5)
        ax3.fill_between(steps, self.step_metrics['total_throughput'], alpha=0.3, color='green')
        ax3.set_xlabel('Step', fontweight='bold')
        ax3.set_ylabel('Throughput (Mbps)', fontweight='bold')
        ax3.set_title(f'Throughput (SE≤{MAX_SPECTRAL_EFFICIENCY})', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#f8f9fa')
        
        # Handovers
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(steps, self.step_metrics['handover_count'], 'r-', linewidth=2.5, alpha=0.7)
        ax4.fill_between(steps, self.step_metrics['handover_count'], alpha=0.3, color='red')
        ax4.set_xlabel('Step', fontweight='bold')
        ax4.set_ylabel('Handovers', fontweight='bold')
        ax4.set_title('Handovers (TTT=5)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('#f8f9fa')
        
        # BS Load
        ax5 = fig.add_subplot(3, 4, 6)
        bs_loads = [bs.load for bs in self.base_stations]
        bs_types = [bs.bs_type for bs in self.base_stations]
        colors = ['#e74c3c' if t == 'macro' else '#f39c12' if t == 'micro' else '#27ae60' 
                 for t in bs_types]
        ax5.bar(range(self.num_bs), bs_loads, color=colors, 
               edgecolor='black', linewidth=1.5)
        ax5.set_xlabel('BS ID', fontweight='bold')
        ax5.set_ylabel('Load (UEs)', fontweight='bold')
        ax5.set_title('BS Load', fontweight='bold')
        ax5.set_xticks(range(self.num_bs))
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_facecolor('#f8f9fa')
        
        # Coverage
        ax6 = fig.add_subplot(3, 4, 7)
        ax6.plot(steps, self.step_metrics['coverage'], 'purple', linewidth=2.5)
        ax6.fill_between(steps, self.step_metrics['coverage'], alpha=0.3, color='purple')
        ax6.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Target')
        ax6.set_xlabel('Step', fontweight='bold')
        ax6.set_ylabel('Coverage (%)', fontweight='bold')
        ax6.set_title('Coverage', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_facecolor('#f8f9fa')
        
        # Regret
        ax7 = fig.add_subplot(3, 4, 8)
        ax7.plot(steps, self.step_metrics['regret'], 'darkred', linewidth=2.5)
        ax7.fill_between(steps, self.step_metrics['regret'], alpha=0.3, color='red')
        ax7.set_xlabel('Step', fontweight='bold')
        ax7.set_ylabel('Cumulative Regret', fontweight='bold')
        ax7.set_title('CMAB Learning', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.set_facecolor('#f8f9fa')
        
        # Remaining plots
        ax8 = fig.add_subplot(3, 4, 9)
        handover_counts = [u.handover_count for u in self.users]
        ax8.hist(handover_counts, bins=range(0, max(handover_counts)+2), 
                color='coral', edgecolor='black', linewidth=1.5, alpha=0.8)
        ax8.set_xlabel('Handovers per UE', fontweight='bold')
        ax8.set_ylabel('Count', fontweight='bold')
        ax8.set_title('Handover Distribution', fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary"""
        print(f"\n{'='*70}")
        print("SIMULATION SUMMARY - COMPREHENSIVELY FIXED")
        print(f"{'='*70}")
        
        print(f"\n--- CRITICAL FIXES ---")
        print(f"[1] SE Cap: {MAX_SPECTRAL_EFFICIENCY} bits/s/Hz")
        print(f"[2] SINR Bounds: [{MIN_SINR_DB}, {MAX_SINR_DB}] dB")
        print(f"[3] TTT: 5 steps before handover")
        print(f"[4] Handover Penalty: 0.2 + 0.15 hysteresis")
        print(f"[5] Correct units (Mbps)")
        
        print(f"\n--- Performance ---")
        print(f"Avg SINR: {np.mean(self.step_metrics['avg_sinr']):.2f} dB")
        print(f"Avg Throughput: {np.mean(self.step_metrics['total_throughput']):.1f} Mbps")
        print(f"Total Handovers: {int(np.sum(self.step_metrics['handover_count']))}")
        print(f"Avg Coverage: {np.mean(self.step_metrics['coverage']):.1f}%")
        
        print(f"\n{'='*70}\n")


# Main execution
if __name__ == "__main__":
    sim = HetNet5GSimulator(
        grid_size=3000,
        num_macro=3,
        num_micro=4,
        num_pico=6,
        num_ues=300,
        tx_power_macro=43,
        noise_power=-174,
        linucb_alpha=1.0
    )
    
    summary = sim.run_simulation(num_steps=200)
    sim.print_summary()
    sim.visualize_results(save_path='5g_hetnet_comprehensively_fixed.png')