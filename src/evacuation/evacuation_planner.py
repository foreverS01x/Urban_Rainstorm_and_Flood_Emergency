import networkx as nx
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import pyproj
from shapely.geometry import Point, LineString
from shapely.ops import transform
from functools import partial
import json

@dataclass
class EvacuationPoint:
    """疏散点数据类"""
    id: str
    location: Point
    capacity: int  # 容纳人数
    current_occupancy: int  # 当前人数
    facility_type: str  # 设施类型（学校、体育场等）
    accessibility: float  # 可达性指数
    risk_level: float  # 风险等级

@dataclass
class RoadSegment:
    """道路段数据类"""
    id: str
    geometry: LineString
    road_type: str  # 道路类型
    width: float  # 道路宽度
    capacity: float  # 通行能力
    current_flow: float  # 当前流量
    risk_level: float  # 风险等级
    status: str  # 道路状态（通畅、拥堵、封闭等）

class EvacuationPlanner:
    """疏散路径规划器"""
    
    def __init__(self):
        self.road_network = nx.DiGraph()  # 道路网络
        self.evacuation_points = {}  # 疏散点
        self.population_clusters = []  # 人群聚集点
        self.risk_areas = None  # 风险区域
        
    def load_road_network(self, road_data: gpd.GeoDataFrame):
        """加载道路网络数据"""
        for idx, row in road_data.iterrows():
            # 创建道路段对象
            segment = RoadSegment(
                id=str(idx),
                geometry=row.geometry,
                road_type=row['road_type'],
                width=row['width'],
                capacity=self._calculate_road_capacity(row),
                current_flow=0,
                risk_level=0,
                status='normal'
            )
            
            # 添加到网络
            start_point = Point(segment.geometry.coords[0])
            end_point = Point(segment.geometry.coords[-1])
            
            self.road_network.add_edge(
                start_point,
                end_point,
                segment=segment,
                weight=segment.geometry.length
            )
    
    def load_evacuation_points(self, points_data: gpd.GeoDataFrame):
        """加载疏散点数据"""
        for idx, row in points_data.iterrows():
            point = EvacuationPoint(
                id=str(idx),
                location=row.geometry,
                capacity=row['capacity'],
                current_occupancy=0,
                facility_type=row['type'],
                accessibility=self._calculate_accessibility(row.geometry),
                risk_level=0
            )
            self.evacuation_points[point.id] = point
    
    def identify_population_clusters(self, population_data: gpd.GeoDataFrame):
        """识别人群聚集点"""
        # 提取人口密度高的区域
        coords = np.vstack((
            population_data.geometry.x,
            population_data.geometry.y
        )).T
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=0.01, min_samples=5).fit(coords)
        
        # 保存聚类结果
        self.population_clusters = []
        for label in set(clustering.labels_):
            if label != -1:  # 排除噪声点
                cluster_points = coords[clustering.labels_ == label]
                cluster_center = cluster_points.mean(axis=0)
                cluster_population = len(cluster_points)
                self.population_clusters.append({
                    'center': Point(cluster_center),
                    'population': cluster_population
                })
    
    def update_risk_levels(self, risk_data: gpd.GeoDataFrame):
        """更新风险等级"""
        self.risk_areas = risk_data
        
        # 更新道路风险等级
        for _, edge in self.road_network.edges(data=True):
            segment = edge['segment']
            risk_level = self._calculate_road_risk(segment.geometry)
            segment.risk_level = risk_level
            edge['weight'] = segment.geometry.length * (1 + risk_level)
        
        # 更新疏散点风险等级
        for point in self.evacuation_points.values():
            point.risk_level = self._calculate_point_risk(point.location)
    
    def update_road_status(self, traffic_data: Dict[str, str]):
        """更新道路状态"""
        for road_id, status in traffic_data.items():
            for _, edge in self.road_network.edges(data=True):
                if edge['segment'].id == road_id:
                    edge['segment'].status = status
                    # 根据状态调整路段权重
                    weight_factors = {
                        'normal': 1.0,
                        'congested': 2.0,
                        'closed': float('inf')
                    }
                    edge['weight'] = (
                        edge['segment'].geometry.length * 
                        weight_factors.get(status, 1.0) *
                        (1 + edge['segment'].risk_level)
                    )
    
    def plan_evacuation_routes(self, 
                             start_point: Point,
                             population: int,
                             max_routes: int = 3
                             ) -> List[Dict]:
        """规划疏散路径"""
        routes = []
        remaining_population = population
        
        # 获取可用的疏散点
        available_points = [
            point for point in self.evacuation_points.values()
            if point.current_occupancy < point.capacity and
            point.risk_level < 0.7
        ]
        
        # 按可达性和剩余容量排序
        available_points.sort(
            key=lambda x: (x.accessibility, x.capacity - x.current_occupancy),
            reverse=True
        )
        
        for point in available_points[:max_routes]:
            if remaining_population <= 0:
                break
                
            try:
                # 使用Dijkstra算法找最短路径
                path = nx.shortest_path(
                    self.road_network,
                    start_point,
                    point.location,
                    weight='weight'
                )
                
                # 计算路径容量
                path_capacity = self._calculate_path_capacity(path)
                # 分配人数
                assigned_population = min(
                    path_capacity,
                    point.capacity - point.current_occupancy,
                    remaining_population
                )
                
                if assigned_population > 0:
                    # 更新疏散点当前人数
                    point.current_occupancy += assigned_population
                    remaining_population -= assigned_population
                    
                    # 构建路径信息
                    route = {
                        'path': path,
                        'evacuation_point': point,
                        'assigned_population': assigned_population,
                        'risk_level': self._calculate_path_risk(path),
                        'length': self._calculate_path_length(path)
                    }
                    routes.append(route)
            
            except nx.NetworkXNoPath:
                continue
        
        return routes
    
    def _calculate_road_capacity(self, road_data: gpd.GeoSeries) -> float:
        """计算道路通行能力"""
        # 基于道路类型和宽度计算通行能力
        capacity_factors = {
            'highway': 2000,  # 每车道每小时
            'primary': 1500,
            'secondary': 1000,
            'tertiary': 500
        }
        base_capacity = capacity_factors.get(road_data['road_type'], 500)
        lanes = max(1, road_data['width'] // 3.5)  # 假设每车道3.5米
        return base_capacity * lanes
    
    def _calculate_accessibility(self, location: Point) -> float:
        """计算位置可达性"""
        # 考虑到周边道路网络密度和连通性
        buffer_distance = 1000  # 1公里缓冲区
        nearby_roads = sum(1 for _, edge in self.road_network.edges(data=True)
                         if edge['segment'].geometry.distance(location) < buffer_distance)
        return min(1.0, nearby_roads / 50)  # 归一化，假设50条道路为最佳
    
    def _calculate_road_risk(self, geometry: LineString) -> float:
        """计算道路风险等级"""
        if self.risk_areas is None:
            return 0.0
        
        # 计算道路与风险区域的交叉情况
        risk_level = 0.0
        for _, risk_area in self.risk_areas.iterrows():
            if geometry.intersects(risk_area.geometry):
                intersection_length = geometry.intersection(risk_area.geometry).length
                risk_level = max(risk_level, 
                               intersection_length / geometry.length * 
                               risk_area['risk_level'])
        return risk_level
    
    def _calculate_point_risk(self, location: Point) -> float:
        """计算点位风险等级"""
        if self.risk_areas is None:
            return 0.0
        
        # 计算点位与风险区域的关系
        risk_level = 0.0
        for _, risk_area in self.risk_areas.iterrows():
            if location.intersects(risk_area.geometry):
                risk_level = max(risk_level, risk_area['risk_level'])
        return risk_level
    
    def _calculate_path_capacity(self, path: List[Point]) -> float:
        """计算路径通行能力"""
        # 取决于路径上的最小通行能力
        min_capacity = float('inf')
        for i in range(len(path) - 1):
            edge = self.road_network[path[i]][path[i+1]]
            min_capacity = min(min_capacity, edge['segment'].capacity)
        return min_capacity
    
    def _calculate_path_risk(self, path: List[Point]) -> float:
        """计算路径风险等级"""
        # 计算加权平均风险
        total_length = 0
        weighted_risk = 0
        for i in range(len(path) - 1):
            edge = self.road_network[path[i]][path[i+1]]
            segment = edge['segment']
            length = segment.geometry.length
            total_length += length
            weighted_risk += length * segment.risk_level
        return weighted_risk / total_length if total_length > 0 else 0
    
    def _calculate_path_length(self, path: List[Point]) -> float:
        """计算路径长度"""
        total_length = 0
        for i in range(len(path) - 1):
            edge = self.road_network[path[i]][path[i+1]]
            total_length += edge['segment'].geometry.length
        return total_length
    
    def export_routes_to_geojson(self, 
                                routes: List[Dict],
                                output_file: str):
        """导出路径为GeoJSON格式"""
        features = []
        
        # 创建坐标转换器（确保输出WGS84坐标）
        project = partial(
            pyproj.transform,
            pyproj.Proj(self.road_network.graph['crs']),
            pyproj.Proj('EPSG:4326')
        )
        
        for idx, route in enumerate(routes):
            # 构建路径几何
            path_coords = []
            for i in range(len(route['path']) - 1):
                edge = self.road_network[route['path'][i]][route['path'][i+1]]
                path_coords.extend(edge['segment'].geometry.coords)
            
            # 转换坐标系
            path_geom = transform(project, LineString(path_coords))
            
            # 创建Feature
            feature = {
                'type': 'Feature',
                'geometry': json.loads(path_geom.json),
                'properties': {
                    'route_id': idx,
                    'evacuation_point_id': route['evacuation_point'].id,
                    'assigned_population': route['assigned_population'],
                    'risk_level': route['risk_level'],
                    'length': route['length']
                }
            }
            features.append(feature)
        
        # 保存为GeoJSON文件
        feature_collection = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(feature_collection, f, ensure_ascii=False, indent=2) 