# ILStudio RL框架集成设计

## 设计原则

1. **足够抽象**：只定义核心接口，不涉及具体实现
2. **通用性**：支持各种RL算法和训练模式
3. **兼容性**：直接使用MetaEnv和MetaPolicy，无需适配
4. **扩展性**：易于扩展支持并行训练和分布式训练
5. **模块化**：奖励函数、配置系统独立可替换

---

## 目录结构

```
ILStudio/
├── rl/                          # 强化学习模块
│   ├── __init__.py
│   ├── base.py                  # RL 基类定义
│   ├── algorithms/              # RL 算法实现
│   │   ├── __init__.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   ├── td3.py
│   │   ├── dpo.py               # Direct Preference Optimization (VLA)
│   │   ├── grpo.py              # Group Relative Policy Optimization (VLA)
│   │   └── reinforce.py
│   ├── buffer/                  # 经验回放
│   │   ├── __init__.py
│   │   ├── base_replay.py       # BaseReplay基类
│   │   ├── memory_replay.py     # 内存存储实现
│   │   ├── rollout_buffer.py   # On-policy Rollout Buffer
│   │   └── priority_buffer.py  # 优先级采样Buffer
│   ├── rewards/                 # 奖励函数模块
│   │   ├── __init__.py
│   │   ├── base_reward.py       # 奖励函数基类
│   │   ├── sparse_reward.py    # 稀疏奖励
│   │   ├── dense_reward.py     # 密集奖励
│   │   ├── learned_reward.py   # 学习的奖励模型
│   │   └── language_reward.py  # 语言条件奖励（VLA）
│   ├── collectors/              # 数据收集器模块（新增）
│   │   ├── __init__.py
│   │   ├── base_collector.py   # Collector基类
│   │   ├── sim_collector.py    # 仿真环境收集器
│   │   └── real_collector.py   # 真实机器人收集器
│   ├── trainers/                # 训练器实现
│   │   ├── __init__.py
│   │   ├── base_trainer.py      # BaseTrainer基类
│   │   ├── simple_trainer.py   # 单机训练器
│   │   ├── parallel_trainer.py # 并行环境训练器
│   │   └── distributed_trainer.py  # 分布式训练器
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       └── data_processor.py   # 数据处理器（对齐ILStudio pipeline）
├── configs/
│   ├── rl/                      # RL 配置（新增）
│   │   ├── ppo.yaml
│   │   ├── sac.yaml
│   │   ├── dpo.yaml
│   │   └── grpo.yaml
│   └── ...
├── train_rl.py                  # RL 训练入口（新增）
└── ...
```

---

## 核心基类设计

### 1. Replay基类 (`BaseReplay`)

**职责：** 存储和管理经验数据（transitions）

**设计理念：**
- **存储原始Meta数据**：在buffer中存储原始的MetaObs和MetaAction，保持数据的原始性
- **完整信息存储**：支持存储MetaObs和MetaAction的所有字段（state、image、raw_lang、state_ee、state_joint、depth、pc等）
- **扩展性**：支持存储额外的自定义字段（value、log_prob、advantage、trajectory_id等），方便后续扩展
- **采样时转换**：采样时通过转换函数对齐ILStudio的data pipeline（normalization等）
- **兼容性**：既保持数据的原始性，又兼容ILStudio的normalization pipeline

**核心接口：**

```python
class BaseReplay:
    """Replay Buffer基类"""
    
    def __init__(
        self,
        capacity: int = 1000000,
        device: Union[str, torch.device] = 'cpu',
        **kwargs
    ):
        """
        初始化Replay Buffer
        
        Args:
            capacity: Buffer容量（最大存储的transition数量）
            device: 数据存储设备（'cpu'或'cuda'，默认'cpu'）
                   - 'cpu': 数据存储在CPU内存中
                   - 'cuda'或'cuda:0': 数据存储在GPU内存中
            **kwargs: 其他初始化参数
        """
        self.capacity = capacity
        self.device = torch.device(device) if isinstance(device, str) else device
    
    def add(self, transition: Dict[str, Any]) -> None:
        """
        添加一个transition到buffer
        
        存储原始Meta数据（MetaObs、MetaAction），不进行任何normalization
        支持存储MetaObs和MetaAction的所有字段，以及额外的自定义信息
        
        Args:
            transition: 包含以下字段的字典：
                - state: MetaObs格式的当前状态（原始数据，包含所有字段）
                - action: MetaAction格式的动作（原始数据，包含所有字段）
                - reward: float奖励
                - next_state: MetaObs格式的下一个状态（原始数据）
                - done: bool是否结束
                - info: 可选，额外信息字典
                - **其他自定义字段**: 可以存储任何额外的信息
        """
        raise NotImplementedError
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        从buffer中采样一个batch（原始数据）
        
        Args:
            batch_size: batch大小
        
        Returns:
            包含原始Meta数据的字典（未经过normalization）
        """
        raise NotImplementedError
    
    def sample_for_training(
        self, 
        batch_size: int,
        data_processor: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        采样并转换为ILStudio训练格式
        
        Args:
            batch_size: batch大小
            data_processor: 可选的数据处理函数，用于对齐ILStudio pipeline
                          - 如果为None，返回原始数据
                          - 如果提供，应该是一个函数：batch -> processed_batch
        
        Returns:
            处理后的batch数据（符合ILStudio训练格式）
        """
        batch = self.sample(batch_size)
        if data_processor is not None:
            batch = data_processor(batch)
        return batch
    
    def __len__(self) -> int:
        """返回buffer当前大小"""
        raise NotImplementedError
    
    def clear(self) -> None:
        """清空buffer"""
        raise NotImplementedError
    
    def save(self, path: str, **kwargs) -> None:
        """
        保存buffer中的数据到文件
        
        Args:
            path: 保存路径（可以是文件路径或目录路径）
            **kwargs: 保存选项
                - format: 保存格式（如'pkl'、'hdf5'、'npz'等，可选）
                - compress: 是否压缩（可选）
        """
        raise NotImplementedError
    
    def load(self, path: str, **kwargs) -> None:
        """
        从文件加载数据到buffer
        
        Args:
            path: 加载路径（可以是文件路径或目录路径）
            **kwargs: 加载选项
                - format: 加载格式（可选，可自动推断）
                - append: 是否追加到现有buffer（默认False，清空后加载）
        """
        raise NotImplementedError
```

---

### 2. 算法基类 (`BaseAlgorithm`)

**职责：** 定义RL算法的核心逻辑

**设计理念：** 参考SKRL的设计，将replay放到算法中，这样：
- 每个算法可以有自己的replay配置
- 一个Trainer可以训练多个不同的算法（每个算法有自己的replay）
- 更灵活，支持多智能体场景

**核心接口：**

```python
class BaseAlgorithm:
    """RL算法基类"""
    
    def __init__(
        self, 
        meta_policy: MetaPolicy,
        replay: Optional[Union[BaseReplay, Dict[str, BaseReplay]]] = None,
        **kwargs
    ):
        """
        初始化算法
        
        Args:
            meta_policy: ILStudio的MetaPolicy实例（必需属性）
            replay: 支持两种格式：
                   - BaseReplay实例：单个replay buffer（所有环境共享）
                   - Dict[str, BaseReplay]：多个replay buffer（按环境类型分离）
                   - None：不使用replay buffer（on-policy算法）
            **kwargs: 算法特定的参数
        """
        self.meta_policy = meta_policy  # 必需属性
        self.replay = replay  # 可选属性（off-policy算法需要）
    
    def update(self, batch: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        使用一个batch的数据更新策略
        
        Args:
            batch: 可选，batch数据
                  - 如果为None且replay存在，从replay采样
                  - 如果提供，直接使用提供的batch
            **kwargs: 更新参数，可以包括：
                     - batch_size: 从replay采样时的batch大小
                     - env_types: 指定从哪些环境类型的replay采样（当replay是Dict[str, BaseReplay]时）
                                 - 例如：['indoor', 'outdoor']
                                 - 如果为None，从所有环境类型的replay采样
                     - env_weights: 不同环境类型的数据权重（当使用多个环境类型的replay时）
                                   - 例如：{'indoor': 0.6, 'outdoor': 0.4}
                                   - 如果为None，使用均匀权重
        
        Returns:
            包含loss、metrics等信息的字典
        
        示例：
            # 场景1：单个replay buffer
            algorithm.update(batch_size=256)
            
            # 场景2：多个replay buffer（按环境类型分离）
            algorithm.update(
                batch_size=256,
                env_types=['indoor', 'outdoor'],
                env_weights={'indoor': 0.6, 'outdoor': 0.4}
            )
        """
        raise NotImplementedError
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算损失（可选，某些算法可能需要）
        
        Args:
            batch: batch数据
        
        Returns:
            损失值
        """
        raise NotImplementedError
    
    def select_action(self, obs: MetaObs, **kwargs) -> MetaAction:
        """
        选择动作（可选，某些算法可能需要）
        
        Args:
            obs: MetaObs格式的observation
            **kwargs: 其他参数（如exploration等）
        
        Returns:
            MetaAction格式的动作
        """
        # 默认使用meta_policy的select_action
        return self.meta_policy.select_action(obs, **kwargs)
    
    def record_transition(
        self,
        state: MetaObs,
        action: MetaAction,
        reward: float,
        next_state: MetaObs,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        记录transition到replay buffer（如果存在）
        
        支持存储完整的MetaObs和MetaAction信息，以及额外的自定义字段
        如果使用多个replay buffer（按环境类型分离），会根据kwargs中的env_type选择对应的replay
        
        Args:
            state: 当前状态（MetaObs，包含所有字段）
            action: 动作（MetaAction，包含所有字段）
            reward: 奖励
            next_state: 下一个状态（MetaObs，包含所有字段）
            done: 是否结束
            info: 额外信息字典
            **kwargs: 其他自定义字段，可以存储任何额外信息
                     - env_type: 环境类型标识（如果replay是Dict[str, BaseReplay]）
                     - 例如：value、log_prob、advantage、trajectory_id等
        """
        if self.replay is not None:
            transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': info or {},
                **kwargs
            }
            
            env_type = kwargs.get('env_type', None)
            
            if isinstance(self.replay, dict):
                if env_type is None:
                    raise ValueError("env_type must be provided when using multiple replay buffers")
                if env_type not in self.replay:
                    raise ValueError(f"env_type '{env_type}' not found in replay buffers")
                self.replay[env_type].add(transition)
            else:
                self.replay.add(transition)
```

---

### 3. 奖励函数基类 (`BaseReward`)

**职责：** 定义奖励函数的接口

**设计理念：**
- **模块化**：奖励函数独立模块，易于替换和扩展
- **可组合**：支持多个奖励函数组合使用
- **语言条件**：支持VLA模型的语言条件奖励

**核心接口：**

```python
class BaseReward:
    """奖励函数基类"""
    
    def __init__(self, **kwargs):
        """
        初始化奖励函数
        
        Args:
            **kwargs: 奖励函数特定的参数
        """
        pass
    
    def compute(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算奖励
        
        Args:
            state: 当前状态（MetaObs）
            action: 动作（MetaAction）
            next_state: 下一个状态（MetaObs）
            env_reward: 环境原始奖励
            info: 额外信息字典
        
        Returns:
            计算后的奖励值
        """
        raise NotImplementedError
    
    def reset(self, **kwargs) -> None:
        """
        重置奖励函数状态（如果需要）
        
        Args:
            **kwargs: 重置参数
        """
        pass
```

**具体实现示例：**

```python
# rl/rewards/sparse_reward.py
class SparseReward(BaseReward):
    """稀疏奖励：只在任务完成时给予奖励"""
    
    def __init__(self, success_reward: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.success_reward = success_reward
    
    def compute(self, state, action, next_state, env_reward, info):
        if info and info.get('success', False):
            return self.success_reward
        return 0.0

# rl/rewards/dense_reward.py
class DenseReward(BaseReward):
    """密集奖励：基于状态距离的奖励"""
    
    def __init__(self, goal_key: str = 'goal', distance_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.goal_key = goal_key
        self.distance_scale = distance_scale
    
    def compute(self, state, action, next_state, env_reward, info):
        # 计算到目标的距离
        if hasattr(state, 'state') and hasattr(next_state, 'state'):
            goal = info.get(self.goal_key, None)
            if goal is not None:
                prev_dist = np.linalg.norm(state.state - goal)
                curr_dist = np.linalg.norm(next_state.state - goal)
                progress = (prev_dist - curr_dist) * self.distance_scale
                return env_reward + progress
        return env_reward

# rl/rewards/language_reward.py
class LanguageReward(BaseReward):
    """语言条件奖励：基于语言指令的奖励（VLA）"""
    
    def __init__(self, reward_model=None, **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model
    
    def compute(self, state, action, next_state, env_reward, info):
        if self.reward_model is None:
            return env_reward
        
        # 使用奖励模型计算语言条件奖励
        if hasattr(state, 'raw_lang') and state.raw_lang is not None:
            lang_reward = self.reward_model.compute(
                state=state,
                action=action,
                next_state=next_state,
                instruction=state.raw_lang
            )
            return env_reward + lang_reward
        return env_reward
```

---

### 4. 数据收集器基类 (`BaseCollector`)

**职责：** 从环境中收集交互数据并存储到replay buffer

**设计理念：**
- **职责分离**：将数据收集逻辑从trainer中分离，使trainer更专注于训练循环协调
- **环境抽象**：支持单个环境、并行环境、多环境类型
- **原始数据存储**：只存储环境原始奖励，不进行奖励函数计算，保证数据的原始性
- **统计信息**：收集并返回episode统计信息

**核心接口：**

```python
class BaseCollector:
    """数据收集器基类"""
    
    def __init__(
        self,
        meta_envs: Union[MetaEnv, List[MetaEnv], Callable, Dict[str, Any]],
        algorithm: BaseAlgorithm,
        **kwargs
    ):
        """
        初始化收集器
        
        Args:
            meta_envs: 支持多种格式：
                      - MetaEnv实例：单个环境
                      - List[MetaEnv]：环境列表（同类型环境）
                      - Callable：环境工厂函数
                      - Dict[str, Any]：多环境配置字典（支持不同类型环境）
            algorithm: BaseAlgorithm实例（必需属性）
                      - 用于选择动作和记录transition
            **kwargs: 收集器特定的参数
        
        注意：Collector只存储环境原始奖励，不进行奖励函数计算
             奖励函数在trainer中用于训练时的奖励计算
        """
        self.meta_envs = meta_envs
        self.algorithm = algorithm
    
    def collect(self, n_steps: int, env_type: Optional[str] = None) -> Dict[str, Any]:
        """
        收集 n_steps 的交互数据
        
        Args:
            n_steps: 收集的步数
            env_type: 可选，环境类型标识（用于多环境场景）
                     - 如果提供，会在record_transition时传入env_type
                     - 用于支持单个算法在多个不同环境的数据存储
        
        Returns:
            包含统计信息的字典，如：
            - episode_rewards: episode奖励列表
            - episode_lengths: episode长度列表
            - total_steps: 总步数
            - env_type_stats: 按环境类型分组的统计信息（如果支持多环境）
        """
        raise NotImplementedError
    
    def reset(self, **kwargs) -> None:
        """
        重置收集器状态（如重置环境）
        
        Args:
            **kwargs: 重置参数
        """
        raise NotImplementedError
```

**具体实现示例：**

```python
# rl/collectors/sim_collector.py
class SimCollector(BaseCollector):
    """仿真环境数据收集器"""
    
    def __init__(
        self,
        meta_envs: Union[MetaEnv, List[MetaEnv]],
        algorithm: BaseAlgorithm,
        n_envs: int = 1,
        **kwargs
    ):
        super().__init__(meta_envs, algorithm, **kwargs)
        self.n_envs = n_envs
        
        # 初始化环境
        if isinstance(meta_envs, list):
            self.envs = meta_envs
        elif isinstance(meta_envs, MetaEnv):
            self.envs = [meta_envs]
        else:
            raise ValueError(f"Unsupported env type: {type(meta_envs)}")
        
        self._last_obs = None
        self._last_dones = None
    
    def reset(self, **kwargs) -> None:
        """重置所有环境"""
        self._last_obs = []
        self._last_dones = []
        for env in self.envs:
            obs = env.reset()
            self._last_obs.append(obs)
            self._last_dones.append(False)
    
    def collect(self, n_steps: int, env_type: Optional[str] = None) -> Dict[str, Any]:
        """
        收集 n_steps 的交互数据
        
        Args:
            n_steps: 收集的步数
            env_type: 可选，环境类型标识（用于多环境场景）
                     - 如果提供，会在record_transition时传入env_type
                     - 用于支持单个算法在多个不同环境的数据存储
        
        Returns:
            统计信息字典
        """
        if self._last_obs is None:
            self.reset()
        
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'total_steps': 0
        }
        
        for step in range(n_steps):
            # 获取动作
            actions = []
            for i, obs in enumerate(self._last_obs):
                if not self._last_dones[i]:
                    with torch.no_grad():
                        action = self.algorithm.select_action(obs)
                        actions.append(action)
                else:
                    # 如果环境已结束，使用dummy action
                    actions.append(None)
            
            # 环境交互
            new_obs_list = []
            rewards = []
            dones = []
            infos = []
            
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                if action is not None:
                    new_obs, reward, done, info = env.step(action)
                    
                    # 记录transition（只存储环境原始奖励，不进行奖励函数计算）
                    # 如果提供了env_type，会在record_transition时传入，用于多环境数据分离
                    transition_kwargs = {}
                    if env_type is not None:
                        transition_kwargs['env_type'] = env_type
                    
                    self.algorithm.record_transition(
                        state=self._last_obs[i],
                        action=action,
                        reward=reward,  # 存储原始奖励
                        next_state=new_obs,
                        done=done,
                        info=info,
                        **transition_kwargs  # 传入env_type等额外信息
                    )
                    
                    new_obs_list.append(new_obs)
                    rewards.append(reward)  # 统计使用原始奖励
                    dones.append(done)
                    infos.append(info)
                    
                    # 统计episode信息
                    if done and 'episode' in info:
                        stats['episode_rewards'].append(info['episode'].get('r', 0))
                        stats['episode_lengths'].append(info['episode'].get('l', 0))
                    
                    # 如果episode结束，重置环境
                    if done:
                        new_obs = env.reset()
                        new_obs_list[i] = new_obs
                else:
                    new_obs_list.append(self._last_obs[i])
                    rewards.append(0)
                    dones.append(True)
                    infos.append({})
            
            self._last_obs = new_obs_list
            self._last_dones = dones
            stats['total_steps'] += len([d for d in dones if not d])
        
        return stats

# rl/collectors/real_collector.py
class RealCollector(BaseCollector):
    """真实机器人数据收集器"""
    
    def __init__(
        self,
        meta_envs: MetaEnv,  # 真实机器人环境
        algorithm: BaseAlgorithm,
        **kwargs
    ):
        super().__init__(meta_envs, algorithm, **kwargs)
        # 真实机器人特定的初始化...
    
    def collect(self, n_steps: int) -> Dict[str, Any]:
        """
        从真实机器人收集数据
        
        注意：真实机器人收集可能需要特殊的安全检查和限制
        """
        # 实现真实机器人数据收集逻辑
        # 可能需要添加安全限制、速度限制等
        pass
```

---

### 5. 训练器基类 (`BaseTrainer`)

**职责：** 协调环境、策略和算法，执行训练循环

**核心接口：**

```python
class BaseTrainer:
    """RL训练器基类"""
    
    def __init__(
        self,
        meta_envs: Union[MetaEnv, List[MetaEnv], Callable, Dict[str, Any]],
        algorithm: Union[BaseAlgorithm, List[BaseAlgorithm]],
        collector: Optional[BaseCollector] = None,
        reward_fn: Optional[Union[BaseReward, Callable]] = None,
        **kwargs
    ):
        """
        初始化训练器
        
        Args:
            meta_envs: 支持多种格式：
                      - MetaEnv实例：单个环境
                      - List[MetaEnv]：环境列表（同类型环境）
                      - Callable：环境工厂函数
                      - Dict[str, Any]：多环境配置字典（支持不同类型环境）
            algorithm: BaseAlgorithm实例或BaseAlgorithm列表（必需属性）
                      - 单个算法：单个智能体训练
                      - 算法列表：多个算法在同一个环境中独立训练（每个算法有自己的replay buffer）
            collector: 可选的数据收集器（如果为None，trainer会创建默认collector）
                      - 单个算法：单个collector
                      - 多个算法：可以是collector列表，每个算法对应一个collector
                      - 如果为None，trainer会为每个算法创建默认collector
            reward_fn: 可选的奖励函数（如果为None，使用环境原始奖励）
                      - 可以是BaseReward实例或Callable函数
                      - 在训练时用于计算奖励（用于算法更新）
                      - 注意：replay buffer中存储的是原始奖励，奖励函数只在训练时应用
            **kwargs: 训练器特定的参数
        """
        self.meta_envs = meta_envs
        self.algorithm = algorithm
        self.reward_fn = reward_fn
        
        # 处理collector初始化
        if collector is None:
            from rl.collectors import SimCollector
            if isinstance(algorithm, BaseAlgorithm):
                # 单个算法：创建单个collector
                collector = SimCollector(
                    meta_envs=meta_envs,
                    algorithm=algorithm
                )
            else:
                # 多个算法：为每个算法创建collector
                collector = [
                    SimCollector(
                        meta_envs=meta_envs,
                        algorithm=alg
                    )
                    for alg in algorithm
                ]
        
        self.collector = collector
    
    def train(self, **kwargs) -> None:
        """
        执行训练循环
        
        Args:
            **kwargs: 训练参数，可以包括：
                - total_steps: 总训练步数（可选）
                - total_episodes: 总episode数（可选）
                - max_time: 最大训练时间（可选）
                - log_interval: 日志记录间隔（可选）
                - save_interval: 模型保存间隔（可选）
                - eval_interval: 评估间隔（可选）
        """
        raise NotImplementedError
    
    def compute_reward(
        self,
        state: MetaObs,
        action: MetaAction,
        next_state: MetaObs,
        env_reward: float,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        计算奖励（支持自定义奖励函数）
        
        在训练时使用，用于算法更新时的奖励计算
        replay buffer中存储的是原始奖励，奖励函数只在训练时应用
        
        Args:
            state: 当前状态
            action: 动作
            next_state: 下一个状态
            env_reward: 环境原始奖励
            info: 额外信息字典
        
        Returns:
            计算后的奖励值
        """
        if self.reward_fn is not None:
            if isinstance(self.reward_fn, BaseReward):
                return self.reward_fn.compute(state, action, next_state, env_reward, info)
            else:
                return self.reward_fn(state, action, next_state, env_reward, info)
        return env_reward
    
    def collect_rollout(self, n_steps: int, env_type: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        收集rollout数据（使用collector）
        
        Args:
            n_steps: 收集步数
            env_type: 可选，环境类型标识（用于多环境场景）
                     - 用于支持单个算法在多个不同环境的数据存储
        
        Returns:
            - 单个算法：rollout统计信息字典
            - 多个算法：rollout统计信息字典列表
        """
        if isinstance(self.collector, list):
            # 多个算法：每个算法独立收集数据
            return [col.collect(n_steps, env_type=env_type) for col in self.collector]
        else:
            # 单个算法
            return self.collector.collect(n_steps, env_type=env_type)
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
        env_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        评估策略性能
        
        Args:
            n_episodes: 评估的episode数量
            render: 是否渲染环境（可选）
            env_type: 可选，指定评估的环境类型
            **kwargs: 其他评估参数
        
        Returns:
            包含评估指标的字典
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """保存模型和训练状态"""
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """加载模型和训练状态"""
        raise NotImplementedError
```

---

## 配置系统

### 配置结构

RL配置遵循ILStudio的配置系统设计，使用YAML格式，支持命令行覆盖。

#### `configs/rl/ppo.yaml`

```yaml
name: ppo
type: rl.algorithms.ppo.PPOAlgorithm

# 算法参数
algorithm:
  gamma: 0.99                    # 折扣因子
  gae_lambda: 0.95               # GAE lambda
  clip_range: 0.2                # PPO clip range
  value_coef: 0.5                # Value loss 系数
  entropy_coef: 0.01             # 熵正则化系数
  max_grad_norm: 0.5             # 梯度裁剪
  n_steps: 2048                  # 每次更新的步数
  batch_size: 64                 # Mini-batch 大小
  n_epochs: 10                    # 每次更新的 epoch 数
  learning_rate: 3e-4

# Replay Buffer配置（on-policy算法使用RolloutBuffer）
replay:
  type: rl.buffer.rollout_buffer.RolloutBuffer
  capacity: 2048
  device: cpu
  n_envs: 8
  gae_lambda: 0.95
  gamma: 0.99

# 奖励函数配置（可选，在trainer中使用，用于训练时的奖励计算）
reward:
  type: rl.rewards.dense_reward.DenseReward
  goal_key: goal
  distance_scale: 1.0
  # 或者使用组合奖励
  # type: rl.rewards.composite_reward.CompositeReward
  # components:
  #   - type: rl.rewards.sparse_reward.SparseReward
  #     success_reward: 1.0
  #   - type: rl.rewards.dense_reward.DenseReward
  #     distance_scale: 0.1

# 数据收集器配置（可选，trainer会创建默认collector）
collector:
  type: rl.collectors.sim_collector.SimCollector
  n_envs: 8

# 策略网络配置（复用现有 policy 配置）
policy:
  type: policy.act  # 或 policy.diffusion_policy, policy.openvla 等
  # 继承对应 policy 的配置...

# 价值网络配置（可选，Actor-Critic算法需要）
value_network:
  type: mlp
  hidden_dims: [256, 256]
  activation: relu

# 训练配置
training:
  total_timesteps: 1000000
  save_freq: 10000
  eval_freq: 5000
  log_interval: 100
```

#### `configs/rl/sac.yaml`

```yaml
name: sac
type: rl.algorithms.sac.SACAlgorithm

# 算法参数
algorithm:
  gamma: 0.99
  tau: 0.005                      # Soft update coefficient
  learning_rate: 3e-4
  batch_size: 256
  target_update_interval: 1
  alpha: 0.2                      # Temperature parameter (auto-tuned if None)

# Replay Buffer配置（off-policy算法使用ReplayBuffer）
replay:
  type: rl.buffer.memory_replay.MemoryReplay
  capacity: 1000000
  device: cpu
  prioritized: false             # 是否使用优先级采样

# 奖励函数配置（可选）
reward:
  type: rl.rewards.dense_reward.DenseReward
  goal_key: goal
  distance_scale: 1.0

# 策略网络配置
policy:
  type: policy.act

# Q网络配置
q_network:
  type: mlp
  hidden_dims: [256, 256]
  activation: relu

# 训练配置
training:
  total_timesteps: 1000000
  save_freq: 10000
  eval_freq: 5000
  log_interval: 100
```

#### `configs/rl/dpo.yaml` (VLA Fine-tuning)

```yaml
name: dpo
type: rl.algorithms.dpo.DPOAlgorithm

# 算法参数
algorithm:
  beta: 0.1                      # KL penalty coefficient
  learning_rate: 1e-5
  batch_size: 32
  reference_free: false         # 是否使用无参考模型的 DPO
  label_smoothing: 0.0

# 策略网络配置（VLA模型）
policy:
  type: policy.openvla
  # 继承对应 policy 的配置...

# 参考策略配置（用于KL散度计算）
reference_policy:
  type: policy.openvla
  # 通常是预训练模型的frozen副本

# 奖励函数配置（语言条件奖励）
reward:
  type: rl.rewards.language_reward.LanguageReward
  reward_model:
    type: learned_reward
    checkpoint: ckpt/reward_model.pth

# 训练配置
training:
  total_episodes: 1000
  save_freq: 100
  eval_freq: 50
  log_interval: 10
```

### 配置加载

```python
# 在ConfigLoader中添加RL配置加载方法
class ConfigLoader:
    # ... 现有方法 ...
    
    def load_rl(self, name_or_path: str) -> Tuple[Dict[str, Any], str]:
        """
        加载RL配置
        
        Args:
            name_or_path: RL配置名称或路径（如'ppo'或'rl/ppo'）
        
        Returns:
            (配置字典, 配置文件路径)
        """
        return self.load_yaml_config('rl', name_or_path)
```

---

## VLA 强化学习特殊处理

### DPO (Direct Preference Optimization)

```python
# rl/algorithms/dpo.py
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from rl.base import BaseAlgorithm
from benchmark.base import MetaPolicy

@dataclass
class DPOConfig:
    """DPO 特定配置"""
    beta: float = 0.1
    reference_free: bool = False
    label_smoothing: float = 0.0
    learning_rate: float = 1e-5
    batch_size: int = 32

class DPOAlgorithm(BaseAlgorithm):
    """
    Direct Preference Optimization for VLA
    
    适用于：
    - 有偏好数据 (chosen vs rejected trajectories)
    - Fine-tuning 预训练 VLA 模型
    """
    
    def __init__(
        self,
        meta_policy: MetaPolicy,
        ref_policy: MetaPolicy,  # 参考策略 (frozen)
        config: DPOConfig,
        **kwargs
    ):
        super().__init__(meta_policy=meta_policy, **kwargs)
        self.ref_policy = ref_policy
        self.config = config
        
        # 冻结参考策略
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
    
    def compute_dpo_loss(
        self,
        chosen_obs: MetaObs,
        chosen_actions: MetaAction,
        rejected_obs: MetaObs,
        rejected_actions: MetaAction
    ) -> Dict[str, torch.Tensor]:
        """
        计算 DPO 损失
        
        L_DPO = -log(σ(β * (log π(a_w|s) - log π_ref(a_w|s) 
                         - log π(a_l|s) + log π_ref(a_l|s))))
        """
        # 计算当前策略的 log prob
        chosen_logps = self.meta_policy.get_log_prob(chosen_obs, chosen_actions)
        rejected_logps = self.meta_policy.get_log_prob(rejected_obs, rejected_actions)
        
        # 计算参考策略的 log prob
        with torch.no_grad():
            ref_chosen_logps = self.ref_policy.get_log_prob(chosen_obs, chosen_actions)
            ref_rejected_logps = self.ref_policy.get_log_prob(rejected_obs, rejected_actions)
        
        # DPO 损失
        chosen_rewards = self.config.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (rejected_logps - ref_rejected_logps)
        
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards.mean(),
            'rejected_rewards': rejected_rewards.mean(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean()
        }
    
    def update(self, batch: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """DPO更新逻辑"""
        if batch is None:
            raise ValueError("DPO requires batch with chosen/rejected pairs")
        
        # 提取chosen和rejected数据
        chosen_obs = batch['chosen_obs']
        chosen_actions = batch['chosen_actions']
        rejected_obs = batch['rejected_obs']
        rejected_actions = batch['rejected_actions']
        
        # 计算损失
        loss_dict = self.compute_dpo_loss(
            chosen_obs, chosen_actions,
            rejected_obs, rejected_actions
        )
        
        # 反向传播
        loss_dict['loss'].backward()
        
        return loss_dict
```

### GRPO (Group Relative Policy Optimization)

```python
# rl/algorithms/grpo.py
from dataclasses import dataclass
import torch
from rl.base import BaseAlgorithm

@dataclass
class GRPOConfig:
    """GRPO 配置"""
    group_size: int = 4
    kl_coef: float = 0.1
    reward_scale: float = 1.0
    learning_rate: float = 1e-5

class GRPOAlgorithm(BaseAlgorithm):
    """
    Group Relative Policy Optimization
    
    适用于 VLA 的在线强化学习：
    1. 对每个任务/指令采样多个轨迹
    2. 使用奖励对轨迹进行排序
    3. 使用组内相对奖励进行策略优化
    """
    
    def __init__(
        self,
        meta_policy: MetaPolicy,
        ref_policy: MetaPolicy,  # 参考策略 (frozen)
        config: GRPOConfig,
        **kwargs
    ):
        super().__init__(meta_policy=meta_policy, **kwargs)
        self.ref_policy = ref_policy
        self.config = config
        
        # 冻结参考策略
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
    
    def collect_group_rollouts(self, obs: MetaObs, n_samples: int) -> List[Dict]:
        """
        对同一观测采样多个动作序列
        
        Args:
            obs: 初始观测
            n_samples: 采样数量
        
        Returns:
            轨迹列表
        """
        trajectories = []
        for _ in range(n_samples):
            traj = self._rollout_episode(obs)
            trajectories.append(traj)
        return trajectories
    
    def compute_grpo_loss(
        self,
        trajectories: List[Dict],
        rewards: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        计算 GRPO 损失
        
        使用组内相对奖励作为 advantage
        """
        # 归一化组内奖励
        rewards_tensor = torch.tensor(rewards)
        normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        loss = 0
        for traj, adv in zip(trajectories, normalized_rewards):
            log_probs = self.meta_policy.get_trajectory_log_prob(traj)
            loss -= (log_probs * adv).mean()
        
        # KL 惩罚
        kl_loss = self._compute_kl_penalty(trajectories)
        
        total_loss = loss + self.config.kl_coef * kl_loss
        
        return {
            'loss': total_loss,
            'policy_loss': loss,
            'kl_loss': kl_loss,
            'mean_reward': rewards_tensor.mean()
        }
    
    def _compute_kl_penalty(self, trajectories: List[Dict]) -> torch.Tensor:
        """计算KL散度惩罚"""
        kl_loss = 0
        for traj in trajectories:
            current_logps = self.meta_policy.get_trajectory_log_prob(traj)
            with torch.no_grad():
                ref_logps = self.ref_policy.get_trajectory_log_prob(traj)
            kl = (current_logps - ref_logps).mean()
            kl_loss += kl
        return kl_loss / len(trajectories)
```

---

## 训练入口 `train_rl.py`

```python
#!/usr/bin/env python3
"""
ILStudio Reinforcement Learning Training Script

支持:
- 传统 RL (PPO, SAC) 训练机器人控制策略
- VLA fine-tuning (DPO, GRPO) 使用强化学习
- 混合训练 (IL + RL)
"""

import argparse
from loguru import logger
from configs.loader import ConfigLoader
from data_utils.utils import set_seed
from policy.policy_loader import PolicyLoader
from rl.algorithms import get_algorithm_class
from rl.buffer import get_replay_class
from rl.rewards import get_reward_class
from rl.collectors import get_collector_class
from rl.trainers import get_trainer_class

def parse_args():
    parser = argparse.ArgumentParser(description='RL Training for ILStudio')
    
    # 基础配置
    parser.add_argument('-p', '--policy', type=str, required=True,
                       help='Policy config (e.g., act, diffusion_policy, openvla)')
    parser.add_argument('-r', '--rl_config', type=str, default='ppo',
                       help='RL algorithm config (e.g., ppo, sac, dpo)')
    parser.add_argument('-e', '--env', type=str, required=True,
                       help='Environment config')
    parser.add_argument('-o', '--output_dir', type=str, default='ckpt/rl_output')
    
    # 训练模式
    parser.add_argument('--mode', type=str, default='online',
                       choices=['online', 'offline', 'hybrid'],
                       help='Training mode: online (env interaction), offline (dataset), hybrid')
    
    # 预训练模型 (用于 fine-tuning)
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    
    # 其他
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    
    args, unknown = parser.parse_known_args()
    args.unknown_args = unknown
    return args


def create_algorithm(rl_config, policy, env_config, args):
    """创建RL算法实例"""
    # 创建replay buffer（如果配置中有）
    replay = None
    if 'replay' in rl_config:
        replay_class = get_replay_class(rl_config['replay']['type'])
        replay = replay_class(**rl_config['replay'])
    
    # 创建算法实例（奖励函数不在算法中，在trainer中）
    algorithm_class = get_algorithm_class(rl_config['type'])
    algorithm = algorithm_class(
        meta_policy=policy,
        replay=replay,
        **rl_config.get('algorithm', {})
    )
    
    return algorithm


def create_reward_fn(rl_config):
    """创建奖励函数实例（在trainer中使用，用于训练时的奖励计算）"""
    reward_fn = None
    if 'reward' in rl_config:
        reward_class = get_reward_class(rl_config['reward']['type'])
        reward_fn = reward_class(**rl_config['reward'])
    return reward_fn


def create_collector(rl_config, env, algorithm):
    """创建数据收集器实例（不包含reward_fn，只存储原始奖励）"""
    collector = None
    if 'collector' in rl_config:
        collector_class = get_collector_class(rl_config['collector']['type'])
        collector = collector_class(
            meta_envs=env,
            algorithm=algorithm,
            **rl_config['collector']
        )
    return collector


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 加载配置
    cfg_loader = ConfigLoader(args=args, unknown_args=args.unknown_args)
    policy_config, _ = cfg_loader.load_policy(args.policy)
    rl_config, _ = cfg_loader.load_rl(args.rl_config)
    env_config, _ = cfg_loader.load_env(args.env)
    
    # 创建环境
    env = create_env_from_config(env_config)
    
    # 加载/创建策略
    policy_loader = PolicyLoader()
    if args.pretrained:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        policy = policy_loader.load_pretrained(args.pretrained, policy_config)
    else:
        logger.info("Creating policy from scratch")
        policy = policy_loader.create_policy(policy_config, args)
    
    # 创建RL算法
    algorithm = create_algorithm(rl_config, policy, env_config, args)
    
    # 创建奖励函数（在trainer中使用，用于训练时的奖励计算）
    reward_fn = create_reward_fn(rl_config)
    
    # 创建数据收集器（可选，trainer会创建默认的，不包含reward_fn）
    collector = create_collector(rl_config, env, algorithm)
    
    # 创建训练器
    trainer_class = get_trainer_class('simple')  # 或从配置中读取
    trainer = trainer_class(
        meta_envs=env,
        algorithm=algorithm,
        collector=collector,  # collector在trainer中（只存储原始奖励）
        reward_fn=reward_fn  # reward_fn在trainer中（用于训练时的奖励计算）
    )
    
    # 训练
    logger.info("="*60)
    logger.info(f"🚀 Starting RL Training: {rl_config['name']}")
    logger.info(f"   Policy: {policy_config.get('name', args.policy)}")
    logger.info(f"   Environment: {env_config.get('name', args.env)}")
    logger.info(f"   Mode: {args.mode}")
    logger.info("="*60)
    
    trainer.train(
        total_steps=rl_config['training']['total_timesteps'],
        log_interval=rl_config['training'].get('log_interval', 100),
        save_interval=rl_config['training'].get('save_freq', 10000),
        eval_interval=rl_config['training'].get('eval_freq', 5000),
        output_dir=args.output_dir
    )
    
    logger.info("✓ Training completed!")


if __name__ == '__main__':
    main()
```

---

## 使用示例

### 示例1：传统RL训练（PPO + ACT）

```bash
python train_rl.py \
    -p act \
    -r ppo \
    -e libero.example \
    -o ckpt/rl_act_ppo
```

### 示例2：VLA Fine-tuning（DPO + OpenVLA）

```bash
python train_rl.py \
    -p openvla \
    -r dpo \
    -e behavior1k.example \
    --pretrained ckpt/openvla_pretrained \
    -o ckpt/openvla_dpo
```

### 示例3：使用自定义奖励函数

```python
# 在配置文件中指定奖励函数
# configs/rl/ppo.yaml
reward:
  type: rl.rewards.composite_reward.CompositeReward
  components:
    - type: rl.rewards.sparse_reward.SparseReward
      success_reward: 1.0
      weight: 0.5
    - type: rl.rewards.dense_reward.DenseReward
      goal_key: goal
      distance_scale: 0.1
      weight: 0.5
```

### 示例4：单个算法在多个不同环境的训练（场景1）

```python
# 场景1：单个算法在多个不同环境（如室内、室外）中训练
# 数据按环境类型分离存储到不同的replay buffer

from rl.buffer import MemoryReplay
from rl.algorithms import SACAlgorithm
from rl.collectors import SimCollector
from rl.trainers import SimpleTrainer

# 创建多个环境的replay buffer（按环境类型分离）
replay = {
    'indoor': MemoryReplay(capacity=1000000, device='cpu'),
    'outdoor': MemoryReplay(capacity=1000000, device='cpu')
}

# 创建算法（使用多个replay buffer）
algorithm = SACAlgorithm(
    meta_policy=policy,
    replay=replay  # Dict[str, BaseReplay]，按环境类型分离
)

# 创建多个环境的collector
indoor_collector = SimCollector(
    meta_envs=indoor_envs,  # 室内环境
    algorithm=algorithm
)

outdoor_collector = SimCollector(
    meta_envs=outdoor_envs,  # 室外环境
    algorithm=algorithm
)

# 创建trainer
trainer = SimpleTrainer(
    meta_envs={'indoor': indoor_envs, 'outdoor': outdoor_envs},
    algorithm=algorithm,
    reward_fn=reward_fn
)

# 训练时，分别从不同环境收集数据
# 数据会自动存储到对应环境类型的replay buffer中
for step in range(total_steps):
    # 从室内环境收集数据（env_type='indoor'）
    indoor_stats = indoor_collector.collect(n_steps=1000, env_type='indoor')
    
    # 从室外环境收集数据（env_type='outdoor'）
    outdoor_stats = outdoor_collector.collect(n_steps=1000, env_type='outdoor')
    
    # 更新算法（可以从不同环境类型的replay混合采样）
    loss = algorithm.update(
        batch_size=256,
        env_types=['indoor', 'outdoor'],  # 指定从哪些环境采样
        env_weights={'indoor': 0.6, 'outdoor': 0.4}  # 不同环境的数据权重
    )
```

### 示例5：多个算法在同一个环境中独立训练（场景2）

```python
# 场景2：多个算法在同一个环境中独立训练
# 每个算法有自己的replay buffer和collector

from rl.buffer import MemoryReplay
from rl.algorithms import SACAlgorithm, PPOAlgorithm
from rl.collectors import SimCollector
from rl.trainers import SimpleTrainer

# 创建多个算法，每个算法有自己的replay buffer
algorithm1 = SACAlgorithm(
    meta_policy=policy1,
    replay=MemoryReplay(capacity=1000000, device='cpu')
)

algorithm2 = PPOAlgorithm(
    meta_policy=policy2,
    replay=MemoryReplay(capacity=100000, device='cpu')
)

# 创建多个collector，每个算法对应一个collector
collector1 = SimCollector(
    meta_envs=env,  # 同一个环境
    algorithm=algorithm1
)

collector2 = SimCollector(
    meta_envs=env,  # 同一个环境
    algorithm=algorithm2
)

# 创建trainer（传入算法列表和collector列表）
trainer = SimpleTrainer(
    meta_envs=env,
    algorithm=[algorithm1, algorithm2],  # 多个算法
    collector=[collector1, collector2],  # 每个算法对应一个collector
    reward_fn=reward_fn
)

# 训练时，每个算法独立收集数据和更新
for step in range(total_steps):
    # 收集rollout（返回每个算法的统计信息）
    stats_list = trainer.collect_rollout(n_steps=1000)
    # stats_list[0] 是 algorithm1 的统计信息
    # stats_list[1] 是 algorithm2 的统计信息
    
    # 每个算法独立更新
    loss1 = algorithm1.update(batch_size=256)
    loss2 = algorithm2.update(batch_size=64)
```

### 示例6：配置文件中支持多环境场景

```yaml
# configs/rl/multi_env_sac.yaml
name: multi_env_sac
type: rl.algorithms.sac.SACAlgorithm

# 算法参数
algorithm:
  gamma: 0.99
  learning_rate: 3e-4
  batch_size: 256

# Replay Buffer配置（按环境类型分离）
replay:
  indoor:
    type: rl.buffer.memory_replay.MemoryReplay
    capacity: 1000000
    device: cpu
  outdoor:
    type: rl.buffer.memory_replay.MemoryReplay
    capacity: 1000000
    device: cpu

# 策略网络配置
policy:
  type: policy.act

# 训练配置
training:
  total_timesteps: 1000000
  env_types: ['indoor', 'outdoor']
  env_weights:
    indoor: 0.6
    outdoor: 0.4
```

---

## 接口关系图

```
┌─────────────────────────────────────────┐
│  BaseReplay                             │
│  (存储经验数据)                          │
│  - 完整MetaObs/MetaAction               │
│  - 自定义字段（value、log_prob等）      │
└─────────────────────────────────────────┘
         ▲
         │ (可选，在算法中)
         │
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ BaseAlgorithm   │      │ BaseCollector   │      │  BaseTrainer    │
│                 │      │                 │      │                 │
│ + meta_policy   │◄─────┤ + algorithm     │◄─────┤ + collector     │
│ + replay        │      │ + meta_envs     │      │ + algorithm     │
│ + update()      │      │ + collect()     │      │ + reward_fn     │
│ + record_       │      │ (只存储原始奖励) │      │ + compute_      │
│   transition()  │      └─────────────────┘      │   reward()      │
└─────────────────┘              │                │ + train()       │
                                 │                │ + evaluate()    │
                                 │                └─────────────────┘
                                 │                        │
                    ┌────────────┼────────────┐         │
                    │            │            │         │ 使用
                    │ 使用       │ 使用       │         │
                    ▼            ▼            ▼         ▼
         ┌──────────┐  ┌──────────┐  ┌─────────────────┐
         │ MetaEnv  │  │MetaPolicy│  │  BaseReward     │
         │          │  │          │  │  (奖励函数)      │
         └──────────┘  └──────────┘  │  - compute()    │
                                     └─────────────────┘
```

---

## 设计亮点

### 1. **完整的Meta数据存储**
- Replay buffer存储完整的MetaObs和MetaAction，包括所有字段
- 不丢失任何信息，方便后续分析和扩展

### 2. **模块化奖励函数**
- 奖励函数独立模块，易于替换和扩展
- 支持组合多个奖励函数
- 支持VLA的语言条件奖励
- **原始数据存储**：replay buffer存储环境原始奖励，奖励函数只在训练时应用
- **数据可复用性**：同样的数据可以用不同的奖励函数进行训练

### 3. **灵活的配置系统**
- 使用YAML配置，支持命令行覆盖
- 复用ILStudio现有的配置加载机制
- 配置驱动，易于实验和部署

### 4. **兼容ILStudio pipeline**
- **原始数据存储**：replay buffer存储原始MetaObs、MetaAction和环境原始奖励
- **奖励函数分离**：奖励函数在trainer中应用，不影响数据收集和存储
- 采样时通过data_processor对齐ILStudio的normalization pipeline
- 既保持灵活性，又兼容现有系统

### 5. **VLA特殊支持**
- 专门的DPO和GRPO算法实现
- 支持语言条件奖励
- 支持参考策略的KL散度约束

### 6. **数据集持久化**
- 支持保存和加载数据集，方便数据管理和复用
- 支持多种保存格式（pkl、hdf5、npz等）
- 支持追加加载，可以合并多个数据集

### 7. **灵活的数据存储策略**
- **场景1：单个算法在多个不同环境**
  - 算法可以使用 `Dict[str, BaseReplay]` 按环境类型分离存储
  - 在 `record_transition` 时传入 `env_type` 参数
  - 支持从不同环境类型的replay混合采样，可配置数据权重
  - 适用于跨域学习、多任务学习等场景

- **场景2：多个算法在同一个环境**
  - Trainer 可以接受 `List[BaseAlgorithm]`，每个算法独立训练
  - 每个算法有自己的 replay buffer 和 collector
  - 支持算法对比、ensemble训练等场景
  - 数据完全隔离，互不干扰

---

## 总结

这个整合设计：

1. ✅ **保持设计2的核心架构**：BaseReplay、BaseAlgorithm、BaseTrainer三个基类
2. ✅ **添加奖励函数模块化**：独立的BaseReward基类和具体实现
3. ✅ **添加配置系统**：YAML配置文件和配置加载机制
4. ✅ **添加VLA支持**：DPO和GRPO算法实现
5. ✅ **添加训练入口**：train_rl.py脚本设计

通过组合这些组件，可以灵活地实现各种RL训练场景，同时保持与ILStudio现有系统的兼容性。

