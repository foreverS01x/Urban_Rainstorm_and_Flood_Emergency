import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from evacuation_planner import EvacuationPlanner

def run_evacuation_example():
    """运行疏散路径规划示例"""
    
    # 1. 创建示例数据
    
    # 道路网络数据
    roads_data = {
        'geometry': [...],  # 道路线数据
        'road_type': ['highway', 'primary', 'secondary', ...],
        'width': [21, 14, 7, ...],  # 道路宽度（米）
    }
    roads_gdf = gpd.GeoDataFrame(roads_data, crs='EPSG:4326')
    
    # 疏散点数据
    evacuation_points_data = {
        'geometry': [...],  # 点位数据
        'capacity': [1000, 800, 500, ...],  # 容纳人数
        'type': ['school', 'stadium', 'park', ...]
    }
    points_gdf = gpd.GeoDataFrame(evacuation_points_data, crs='EPSG:4326')
    
    # 人口分布数据
    population_data = {
        'geometry': [...],  # 点位数据
        'population': [100, 150, 80, ...]
    }
    population_gdf = gpd.GeoDataFrame(population_data, crs='EPSG:4326')
    
    # 风险区域数据
    risk_areas_data = {
        'geometry': [...],  # 多边形数据
        'risk_level': [0.8, 0.6, 0.4, ...]  # 风险等级 (0-1)
    }
    risk_gdf = gpd.GeoDataFrame(risk_areas_data, crs='EPSG:4326')
    
    # 2. 初始化疏散规划器
    planner = EvacuationPlanner()
    
    # 3. 加载数据
    planner.load_road_network(roads_gdf)
    planner.load_evacuation_points(points_gdf)
    planner.identify_population_clusters(population_gdf)
    planner.update_risk_levels(risk_gdf)
    
    # 4. 更新实时路况
    traffic_status = {
        'road1': 'normal',
        'road2': 'congested',
        'road3': 'closed'
    }
    planner.update_road_status(traffic_status)
    
    # 5. 为每个人群聚集点规划疏散路径
    all_routes = []
    for cluster in planner.population_clusters:
        routes = planner.plan_evacuation_routes(
            start_point=cluster['center'],
            population=cluster['population'],
            max_routes=3
        )
        all_routes.extend(routes)
    
    # 6. 导出结果
    planner.export_routes_to_geojson(all_routes, 'evacuation_routes.geojson')
    
    # 7. 打印统计信息
    print("\n疏散路径规划结果：")
    print(f"总人群聚集点数量：{len(planner.population_clusters)}")
    print(f"生成疏散路径数量：{len(all_routes)}")
    
    total_population = sum(cluster['population'] 
                         for cluster in planner.population_clusters)
    evacuated_population = sum(route['assigned_population'] 
                             for route in all_routes)
    
    print(f"总人口：{total_population}")
    print(f"已安排疏散人口：{evacuated_population}")
    print(f"疏散覆盖率：{evacuated_population/total_population*100:.2f}%")
    
    # 打印各疏散点使用情况
    print("\n疏散点使用情况：")
    for point_id, point in planner.evacuation_points.items():
        usage_rate = point.current_occupancy / point.capacity * 100
        print(f"疏散点 {point_id}:")
        print(f"  类型：{point.facility_type}")
        print(f"  容量：{point.capacity}")
        print(f"  当前人数：{point.current_occupancy}")
        print(f"  使用率：{usage_rate:.2f}%")
        print(f"  风险等级：{point.risk_level:.2f}")

if __name__ == '__main__':
    run_evacuation_example() 