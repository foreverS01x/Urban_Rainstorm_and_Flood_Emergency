from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
from typing import List, Tuple, Optional, Dict

class CitizenAgent(Agent):
    """市民智能体"""
    def __init__(self, unique_id: int, model: Model, pos: Tuple[int, int],
                 risk_awareness: float = 0.5):
        super().__init__(unique_id, model)
        self.pos = pos
        self.risk_awareness = risk_awareness
        self.evacuated = False
        self.helped_others = 0
    
    def step(self):
        """智能体行为决策"""
        # 获取当前位置的洪水风险
        current_risk = self.model.get_flood_risk(self.pos)
        
        # 根据风险意识和当前风险决定是否撤离
        if current_risk > (1 - self.risk_awareness):
            self.evacuate()
        else:
            self.help_others()
    
    def evacuate(self):
        """撤离到安全地点"""
        if not self.evacuated:
            safe_pos = self.model.find_safe_location(self.pos)
            if safe_pos:
                self.model.grid.move_agent(self, safe_pos)
                self.evacuated = True
    
    def help_others(self):
        """帮助周围的其他市民"""
        if self.evacuated:
            return
        
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        
        for neighbor in neighbors:
            if isinstance(neighbor, CitizenAgent) and not neighbor.evacuated:
                if self.model.get_flood_risk(neighbor.pos) > 0.7:
                    neighbor.risk_awareness += 0.2
                    self.helped_others += 1

class RescueTeamAgent(Agent):
    """救援队伍智能体"""
    def __init__(self, unique_id: int, model: Model, pos: Tuple[int, int],
                 capacity: int = 5):
        super().__init__(unique_id, model)
        self.pos = pos
        self.capacity = capacity
        self.rescued_count = 0
    
    def step(self):
        """救援决策和行动"""
        if self.rescued_count >= self.capacity:
            self.return_to_base()
        else:
            self.search_and_rescue()
    
    def search_and_rescue(self):
        """搜索和救援需要帮助的市民"""
        # 获取高风险区域的市民
        citizens_in_danger = self.model.get_citizens_in_danger()
        if citizens_in_danger:
            # 选择最近的市民进行救援
            target = min(citizens_in_danger,
                        key=lambda x: self.get_distance(x.pos))
            self.rescue_citizen(target)
    
    def rescue_citizen(self, citizen: CitizenAgent):
        """救援市民"""
        if not citizen.evacuated:
            safe_pos = self.model.find_safe_location(citizen.pos)
            if safe_pos:
                self.model.grid.move_agent(citizen, safe_pos)
                citizen.evacuated = True
                self.rescued_count += 1
    
    def get_distance(self, pos: Tuple[int, int]) -> float:
        """计算到目标位置的距离"""
        return np.sqrt(
            (self.pos[0] - pos[0])**2 + (self.pos[1] - pos[1])**2
        )

class FloodDisasterModel(Model):
    """洪涝灾害模拟模型"""
    def __init__(self, width: int = 50, height: int = 50,
                 num_citizens: int = 100, num_rescue_teams: int = 5,
                 flood_scenario: Optional[np.ndarray] = None):
        super().__init__()
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # 初始化洪水场景
        self.flood_risk = flood_scenario if flood_scenario is not None \
            else np.random.random((width, height))
        
        # 创建市民智能体
        for i in range(num_citizens):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            citizen = CitizenAgent(i, self, (x, y))
            self.grid.place_agent(citizen, (x, y))
            self.schedule.add(citizen)
        
        # 创建救援队伍智能体
        for i in range(num_rescue_teams):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            team = RescueTeamAgent(num_citizens + i, self, (x, y))
            self.grid.place_agent(team, (x, y))
            self.schedule.add(team)
        
        # 数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Evacuated": lambda m: self.get_evacuated_count(),
                "Rescued": lambda m: self.get_rescued_count(),
                "CitizenPositions": lambda m: self.get_citizen_positions(),
                "RescueTeamPositions": lambda m: self.get_rescue_team_positions(),
                "FloodRisk": lambda m: self.flood_risk.tolist()
            }
        )
    
    def step(self):
        """模型单步运行"""
        self.datacollector.collect(self)
        self.schedule.step()
        self.update_flood_risk()
    
    def get_flood_risk(self, pos: Tuple[int, int]) -> float:
        """获取指定位置的洪水风险"""
        return self.flood_risk[pos[0]][pos[1]]
    
    def find_safe_location(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """寻找安全位置"""
        x, y = pos
        search_radius = 5
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                new_x = (x + dx) % self.grid.width
                new_y = (y + dy) % self.grid.height
                if self.flood_risk[new_x][new_y] < 0.3:
                    return (new_x, new_y)
        return None
    
    def get_citizens_in_danger(self) -> List[CitizenAgent]:
        """获取处于危险中的市民"""
        citizens_in_danger = []
        for agent in self.schedule.agents:
            if isinstance(agent, CitizenAgent) and not agent.evacuated:
                if self.get_flood_risk(agent.pos) > 0.7:
                    citizens_in_danger.append(agent)
        return citizens_in_danger
    
    def update_flood_risk(self):
        """更新洪水风险场景"""
        # 简单的扩散模型
        new_risk = self.flood_risk.copy()
        for i in range(1, self.grid.width - 1):
            for j in range(1, self.grid.height - 1):
                new_risk[i][j] = np.mean([
                    self.flood_risk[i-1][j],
                    self.flood_risk[i+1][j],
                    self.flood_risk[i][j-1],
                    self.flood_risk[i][j+1]
                ])
        self.flood_risk = new_risk
    
    def get_evacuated_count(self) -> int:
        """获取已撤离的市民数量"""
        return sum(1 for agent in self.schedule.agents
                  if isinstance(agent, CitizenAgent) and agent.evacuated)
    
    def get_rescued_count(self) -> int:
        """获取被救援的市民数量"""
        return sum(agent.rescued_count for agent in self.schedule.agents
                  if isinstance(agent, RescueTeamAgent))
    
    def get_citizen_positions(self) -> List[Dict]:
        """获取所有市民的位置信息"""
        positions = []
        for agent in self.schedule.agents:
            if isinstance(agent, CitizenAgent):
                positions.append({
                    'id': agent.unique_id,
                    'x': agent.pos[0],
                    'y': agent.pos[1],
                    'evacuated': agent.evacuated,
                    'helped_others': agent.helped_others
                })
        return positions
    
    def get_rescue_team_positions(self) -> List[Dict]:
        """获取所有救援队的位置信息"""
        positions = []
        for agent in self.schedule.agents:
            if isinstance(agent, RescueTeamAgent):
                positions.append({
                    'id': agent.unique_id,
                    'x': agent.pos[0],
                    'y': agent.pos[1],
                    'rescued_count': agent.rescued_count
                })
        return positions 