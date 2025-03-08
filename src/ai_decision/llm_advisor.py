import openai
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv

@dataclass
class EmergencyScenario:
    """应急场景数据类"""
    flood_level: float  # 洪水等级 (0-1)
    affected_population: int  # 受影响人口
    available_resources: Dict[str, int]  # 可用资源
    infrastructure_status: Dict[str, float]  # 基础设施状态
    weather_forecast: Dict[str, float]  # 天气预报

class LLMEmergencyAdvisor:
    """基于大模型的应急决策顾问"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
        
        openai.api_key = self.api_key
        
        self.scenario_template = """
        当前洪涝灾害情况：
        - 洪水等级：{flood_level:.2f}
        - 受影响人口：{affected_population}人
        - 可用资源：{resources}
        - 基础设施状态：{infrastructure}
        - 天气预报：{weather}
        
        请基于以上情况，生成详细的应急响应建议，包括：
        1. 风险评估
        2. 资源调配建议
        3. 撤离路线规划
        4. 救援优先级
        5. 通信策略
        """
    
    async def get_emergency_advice(self, scenario: EmergencyScenario) -> Dict:
        """获取应急建议"""
        prompt = self.scenario_template.format(
            flood_level=scenario.flood_level,
            affected_population=scenario.affected_population,
            resources=json.dumps(scenario.available_resources, ensure_ascii=False),
            infrastructure=json.dumps(scenario.infrastructure_status, 
                                   ensure_ascii=False),
            weather=json.dumps(scenario.weather_forecast, ensure_ascii=False)
        )
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是一个专业的防洪抢险专家，"
                     "请基于给定场景提供专业的应急响应建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return self._parse_llm_response(response.choices[0].message.content)
        except Exception as e:
            print(f"获取LLM建议时出错: {str(e)}")
            return self._get_fallback_advice(scenario)
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析LLM响应"""
        # 这里可以添加更复杂的解析逻辑
        sections = response.split("\n\n")
        parsed_response = {}
        
        for section in sections:
            if section.startswith("1. 风险评估"):
                parsed_response["risk_assessment"] = section
            elif section.startswith("2. 资源调配"):
                parsed_response["resource_allocation"] = section
            elif section.startswith("3. 撤离路线"):
                parsed_response["evacuation_routes"] = section
            elif section.startswith("4. 救援优先级"):
                parsed_response["rescue_priorities"] = section
            elif section.startswith("5. 通信策略"):
                parsed_response["communication_strategy"] = section
        
        return parsed_response
    
    def _get_fallback_advice(self, scenario: EmergencyScenario) -> Dict:
        """获取备用建议"""
        return {
            "risk_assessment": "立即启动应急响应预案",
            "resource_allocation": "优先调配救援队伍和物资",
            "evacuation_routes": "使用预设的安全撤离路线",
            "rescue_priorities": "优先救援高风险区域",
            "communication_strategy": "启动应急广播系统"
        }

class ReinforcementLearningOptimizer:
    """基于强化学习的应急方案优化器"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # 这里可以添加深度学习模型的初始化
        # 例如使用TensorFlow或PyTorch构建Q-network
    
    def get_state_representation(self, scenario: EmergencyScenario) -> np.ndarray:
        """将场景转换为状态向量"""
        state = np.array([
            scenario.flood_level,
            scenario.affected_population / 10000,  # 归一化
            sum(scenario.available_resources.values()) / 100,
            np.mean(list(scenario.infrastructure_status.values())),
            np.mean(list(scenario.weather_forecast.values()))
        ])
        return state
    
    def select_action(self, state: np.ndarray) -> int:
        """选择行动"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # 这里应该使用训练好的模型预测行动
        # 示例：return np.argmax(self.model.predict(state))
        return 0
    
    def train(self, state: np.ndarray, action: int, 
             reward: float, next_state: np.ndarray):
        """训练模型"""
        self.memory.append((state, action, reward, next_state))
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 这里应该添加实际的模型训练代码
        # 例如使用经验回放进行批量训练
    
    def optimize_emergency_plan(self, scenario: EmergencyScenario,
                              current_plan: Dict) -> Dict:
        """优化应急方案"""
        state = self.get_state_representation(scenario)
        action = self.select_action(state)
        
        # 根据选择的行动修改应急方案
        optimized_plan = current_plan.copy()
        
        # 这里应该根据action的值来具体修改方案
        # 示例修改：
        if action == 0:  # 增加资源分配
            if "resource_allocation" in optimized_plan:
                optimized_plan["resource_allocation"] += "\n增加救援队伍部署"
        
        return optimized_plan 