import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Optional

class FloodRiskAnalyzer:
    """洪涝风险分析器"""
    
    def __init__(self):
        self.data = None
        self.cvi_weights = {
            'elevation': 0.2,
            'slope': 0.15,
            'drainage': 0.2,
            'imperviousness': 0.15,
            'historical_floods': 0.3
        }
    
    def load_data(self, gis_data_path: str) -> None:
        """加载GIS数据"""
        self.data = gpd.read_file(gis_data_path)
    
    def calculate_cvi(self) -> pd.Series:
        """计算海岸脆弱性指数(Coastal Vulnerability Index)"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 标准化各指标
        scaler = MinMaxScaler()
        normalized_data = {}
        
        for factor, weight in self.cvi_weights.items():
            if factor in self.data.columns:
                normalized_data[factor] = scaler.fit_transform(
                    self.data[factor].values.reshape(-1, 1)
                ).flatten()
        
        # 计算CVI
        cvi = np.zeros(len(self.data))
        for factor, weight in self.cvi_weights.items():
            if factor in normalized_data:
                cvi += normalized_data[factor] * weight
        
        return pd.Series(cvi, index=self.data.index)
    
    def analyze_vulnerability(self, 
                            additional_factors: Optional[Dict[str, float]] = None
                            ) -> pd.DataFrame:
        """分析区域易损性"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 合并额外因素
        factors = self.cvi_weights.copy()
        if additional_factors:
            factors.update(additional_factors)
        
        # 计算综合易损性指数
        vulnerability_scores = pd.DataFrame()
        
        # 基础易损性（CVI）
        vulnerability_scores['cvi'] = self.calculate_cvi()
        
        # 社会经济因素
        if 'population_density' in self.data.columns:
            vulnerability_scores['social_vulnerability'] = \
                MinMaxScaler().fit_transform(
                    self.data['population_density'].values.reshape(-1, 1)
                )
        
        # 基础设施因素
        if 'infrastructure_quality' in self.data.columns:
            vulnerability_scores['infrastructure_vulnerability'] = \
                MinMaxScaler().fit_transform(
                    self.data['infrastructure_quality'].values.reshape(-1, 1)
                )
        
        return vulnerability_scores
    
    def generate_risk_map(self, output_path: str) -> None:
        """生成风险地图"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 计算综合风险评分
        risk_scores = self.analyze_vulnerability()
        
        # 添加风险评分到GIS数据
        risk_map = self.data.copy()
        risk_map['risk_score'] = risk_scores.mean(axis=1)
        
        # 保存风险地图
        risk_map.to_file(output_path) 