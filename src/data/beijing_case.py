import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta

class BeijingCaseStudy:
    """北京市洪涝风险案例研究数据处理类"""
    
    def __init__(self, data_dir: str = "data/beijing"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.dem_data = None  # 高程数据
        self.landuse_data = None  # 土地利用数据
        self.drainage_network = None  # 排水管网
        self.river_system = None  # 河流水系
        self.historical_floods = None  # 历史洪涝
        self.population = None  # 人口分布
        self.infrastructure = None  # 基础设施
        
    def load_geographic_data(self):
        """加载基础地理数据"""
        try:
            # 加载DEM数据
            with rasterio.open(self.data_dir / "dem.tif") as src:
                self.dem_data = src.read(1)
            
            # 加载土地利用数据
            self.landuse_data = gpd.read_file(self.data_dir / "landuse.shp")
            
            # 加载排水管网
            self.drainage_network = gpd.read_file(self.data_dir / "drainage.shp")
            
            # 加载河流水系
            self.river_system = gpd.read_file(self.data_dir / "rivers.shp")
            
        except Exception as e:
            print(f"加载地理数据时出错: {str(e)}")
    
    def load_historical_data(self):
        """加载历史灾害数据"""
        try:
            # 加载历史积水点
            flood_points = pd.read_csv(self.data_dir / "historical_floods.csv")
            self.historical_floods = gpd.GeoDataFrame(
                flood_points,
                geometry=gpd.points_from_xy(
                    flood_points.longitude, 
                    flood_points.latitude
                )
            )
        except Exception as e:
            print(f"加载历史数据时出错: {str(e)}")
    
    def fetch_weather_data(self, start_date: datetime, end_date: datetime):
        """获取气象数据"""
        # 这里应该实现与气象数据API的接口
        # 例如：中国天气网、气象局API等
        pass
    
    def load_socioeconomic_data(self):
        """加载社会经济数据"""
        try:
            # 加载人口分布数据
            self.population = gpd.read_file(self.data_dir / "population.shp")
            
            # 加载基础设施数据
            self.infrastructure = gpd.read_file(
                self.data_dir / "infrastructure.shp"
            )
        except Exception as e:
            print(f"加载社会经济数据时出错: {str(e)}")
    
    def calculate_risk_factors(self) -> Dict[str, np.ndarray]:
        """计算风险因子"""
        risk_factors = {}
        
        # 1. 地形因子
        if self.dem_data is not None:
            # 计算坡度
            dx, dy = np.gradient(self.dem_data)
            slope = np.sqrt(dx**2 + dy**2)
            risk_factors['terrain'] = self._normalize(slope)
        
        # 2. 土地利用因子
        if self.landuse_data is not None:
            # 计算不透水面积比例
            imperviousness = self._calculate_imperviousness()
            risk_factors['landuse'] = imperviousness
        
        # 3. 历史灾害因子
        if self.historical_floods is not None:
            historical_risk = self._calculate_historical_risk()
            risk_factors['historical'] = historical_risk
        
        # 4. 人口暴露度
        if self.population is not None:
            population_exposure = self._calculate_population_exposure()
            risk_factors['population'] = population_exposure
        
        return risk_factors
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """数据归一化"""
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def _calculate_imperviousness(self) -> np.ndarray:
        """计算不透水面积比例"""
        # 实现不透水面积计算逻辑
        pass
    
    def _calculate_historical_risk(self) -> np.ndarray:
        """计算历史灾害风险"""
        # 实现历史灾害风险计算逻辑
        pass
    
    def _calculate_population_exposure(self) -> np.ndarray:
        """计算人口暴露度"""
        # 实现人口暴露度计算逻辑
        pass
    
    def generate_risk_map(self, output_file: str):
        """生成风险地图"""
        # 计算风险因子
        risk_factors = self.calculate_risk_factors()
        
        # 设置权重
        weights = {
            'terrain': 0.2,
            'landuse': 0.3,
            'historical': 0.3,
            'population': 0.2
        }
        
        # 计算综合风险
        total_risk = np.zeros_like(self.dem_data)
        for factor, weight in weights.items():
            if factor in risk_factors:
                total_risk += risk_factors[factor] * weight
        
        # 保存风险地图
        with rasterio.open(
            self.data_dir / output_file,
            'w',
            driver='GTiff',
            height=total_risk.shape[0],
            width=total_risk.shape[1],
            count=1,
            dtype=total_risk.dtype,
            crs='EPSG:4326'  # WGS84坐标系
        ) as dst:
            dst.write(total_risk, 1)
    
    def export_to_geojson(self, output_file: str):
        """导出数据为GeoJSON格式"""
        features = {
            'type': 'FeatureCollection',
            'features': []
        }
        
        # 添加历史积水点
        if self.historical_floods is not None:
            for _, row in self.historical_floods.iterrows():
                feature = {
                    'type': 'Feature',
                    'geometry': json.loads(row.geometry.json),
                    'properties': {
                        'type': 'flood_point',
                        'date': row.date,
                        'depth': row.depth
                    }
                }
                features['features'].append(feature)
        
        # 添加基础设施
        if self.infrastructure is not None:
            for _, row in self.infrastructure.iterrows():
                feature = {
                    'type': 'Feature',
                    'geometry': json.loads(row.geometry.json),
                    'properties': {
                        'type': 'infrastructure',
                        'name': row.name,
                        'category': row.category
                    }
                }
                features['features'].append(feature)
        
        # 保存为GeoJSON文件
        with open(self.data_dir / output_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2) 