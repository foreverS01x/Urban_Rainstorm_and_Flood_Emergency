from flask import Blueprint, jsonify, request
from src.evacuation.evacuation_planner import EvacuationPlanner
from src.simulation.flood_abm import FloodDisasterModel
from src.ai_decision.llm_advisor import LLMEmergencyAdvisor
from shapely.geometry import Point, LineString
import geopandas as gpd
import numpy as np
import asyncio

api = Blueprint('api', __name__)
planner = EvacuationPlanner()
llm_advisor = LLMEmergencyAdvisor()

# 模拟数据 - 实际项目中应从数据库获取
SAMPLE_CLUSTERS = [
    {
        'id': '1',
        'lat': 39.9042,
        'lng': 116.4074,
        'population': 500
    },
    {
        'id': '2',
        'lat': 39.9142,
        'lng': 116.4174,
        'population': 300
    },
    {
        'id': '3',
        'lat': 39.8942,
        'lng': 116.3974,
        'population': 400
    }
]

SAMPLE_EVACUATION_POINTS = [
    {
        'id': '1',
        'lat': 39.9142,
        'lng': 116.4274,
        'facility_type': '学校',
        'capacity': 1000,
        'current_occupancy': 0
    },
    {
        'id': '2',
        'lat': 39.8942,
        'lng': 116.4174,
        'facility_type': '体育场',
        'capacity': 2000,
        'current_occupancy': 0
    },
    {
        'id': '3',
        'lat': 39.9242,
        'lng': 116.3974,
        'facility_type': '公园',
        'capacity': 1500,
        'current_occupancy': 0
    }
]

SAMPLE_RISK_AREAS = [
    {
        'coordinates': [
            [39.9042, 116.4074],
            [39.9142, 116.4174],
            [39.9042, 116.4274],
            [39.8942, 116.4174]
        ],
        'risk_level': 0.8
    },
    {
        'coordinates': [
            [39.9242, 116.3974],
            [39.9342, 116.4074],
            [39.9242, 116.4174],
            [39.9142, 116.4074]
        ],
        'risk_level': 0.6
    }
]

@api.route('/population_clusters', methods=['GET'])
def get_population_clusters():
    """获取人群聚集点数据"""
    return jsonify({
        'status': 'success',
        'data': SAMPLE_CLUSTERS
    })

@api.route('/evacuation_points', methods=['GET'])
def get_evacuation_points():
    """获取疏散点数据"""
    return jsonify({
        'status': 'success',
        'data': SAMPLE_EVACUATION_POINTS
    })

@api.route('/risk_areas', methods=['GET'])
def get_risk_areas():
    """获取风险区域数据"""
    return jsonify({
        'status': 'success',
        'data': SAMPLE_RISK_AREAS
    })

@api.route('/plan_evacuation', methods=['POST'])
def plan_evacuation():
    """规划疏散路径"""
    data = request.get_json()
    start_point_id = data.get('start_point_id')
    population = data.get('population')
    max_routes = data.get('max_routes', 3)

    # 获取起点数据
    start_cluster = next(
        (c for c in SAMPLE_CLUSTERS if c['id'] == start_point_id),
        None
    )
    if not start_cluster:
        return jsonify({
            'status': 'error',
            'message': '未找到指定的起点'
        }), 400

    # 创建路网数据
    roads_data = create_sample_road_network()
    planner.load_road_network(roads_data)

    # 加载疏散点
    points_data = create_evacuation_points_data()
    planner.load_evacuation_points(points_data)

    # 更新风险等级
    risk_data = create_risk_areas_data()
    planner.update_risk_levels(risk_data)

    # 规划路径
    start_point = Point(start_cluster['lng'], start_cluster['lat'])
    routes = planner.plan_evacuation_routes(
        start_point=start_point,
        population=population,
        max_routes=max_routes
    )

    # 转换路径数据为前端所需格式
    routes_data = []
    for i, route in enumerate(routes):
        coordinates = []
        for point in route['path']:
            coordinates.append([point.y, point.x])  # 转换为[lat, lng]格式

        routes_data.append({
            'coordinates': coordinates,
            'evacuation_point_id': route['evacuation_point'].id,
            'assigned_population': route['assigned_population'],
            'length': route['length'] / 1000,  # 转换为千米
            'risk_level': route['risk_level']
        })

    # 统计信息
    stats = {
        'total_clusters': len(SAMPLE_CLUSTERS),
        'total_routes': len(routes),
        'total_population': population,
        'evacuated_population': sum(r['assigned_population'] for r in routes)
    }

    return jsonify({
        'status': 'success',
        'routes': routes_data,
        'stats': stats
    })

@api.route('/update_road_status', methods=['POST'])
def update_road_status():
    """更新道路状态"""
    data = request.get_json()
    road_id = data.get('road_id')
    status = data.get('status')

    if not road_id or not status:
        return jsonify({
            'status': 'error',
            'message': '缺少必要参数'
        }), 400

    # 更新道路状态
    traffic_data = {road_id: status}
    planner.update_road_status(traffic_data)

    return jsonify({
        'status': 'success',
        'message': '道路状态已更新'
    })

@api.route('/run_simulation', methods=['POST'])
def run_simulation():
    """运行洪水灾害智能体模拟"""
    try:
        data = request.get_json()
        model = FloodDisasterModel(
            width=data.get('width', 50),
            height=data.get('height', 50),
            num_citizens=data.get('num_citizens', 100),
            num_rescue_teams=data.get('num_rescue_teams', 5)
        )

        # 运行模拟并收集数据
        simulation_data = []
        for _ in range(data.get('steps', 100)):
            model.step()
            step_data = {
                'CitizenPositions': [
                    {
                        'x': agent.pos[0],
                        'y': agent.pos[1],
                        'evacuated': agent.evacuated
                    }
                    for agent in model.schedule.agents
                    if agent.type == 'citizen'
                ],
                'RescueTeamPositions': [
                    {
                        'x': agent.pos[0],
                        'y': agent.pos[1]
                    }
                    for agent in model.schedule.agents
                    if agent.type == 'rescue_team'
                ],
                'FloodLevel': model.flood_level,
                'EvacuatedCount': model.evacuated_count,
                'RescuedCount': model.rescued_count
            }
            simulation_data.append(step_data)

        return jsonify({
            'status': 'success',
            'simulation_data': simulation_data,
            'grid_size': {
                'width': model.grid.width,
                'height': model.grid.height
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/get_emergency_advice', methods=['POST'])
async def get_emergency_advice():
    """获取LLM应急决策建议"""
    try:
        data = request.get_json()
        
        # 构建场景描述
        scenario = {
            'flood_level': data['flood_level'],
            'affected_population': data['affected_population'],
            'available_resources': data['available_resources'],
            'infrastructure_status': data['infrastructure_status'],
            'weather_forecast': data['weather_forecast']
        }
        
        # 获取LLM建议
        advice = await llm_advisor.get_advice(scenario)
        
        return jsonify({
            'status': 'success',
            'advice': {
                'risk_assessment': advice['risk_assessment'],
                'resource_allocation': advice['resource_allocation'],
                'evacuation_strategy': advice['evacuation_strategy'],
                'rescue_priorities': advice['rescue_priorities'],
                'communication_plan': advice['communication_plan']
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/get_real_time_data', methods=['GET'])
def get_real_time_data():
    """获取实时监测数据"""
    try:
        # 这里应该连接实际的数据源
        # 示例数据
        data = {
            'water_levels': {
                'station1': 2.5,
                'station2': 3.1,
                'station3': 1.8
            },
            'rainfall': {
                'station1': 25.4,
                'station2': 31.2,
                'station3': 18.7
            },
            'alerts': [
                {
                    'level': 'warning',
                    'location': '区域A',
                    'message': '水位接近警戒线'
                }
            ]
        }
        
        return jsonify({
            'status': 'success',
            'data': data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api.route('/analyze_risk', methods=['POST'])
def analyze_risk():
    """分析区域风险等级"""
    try:
        data = request.get_json()
        
        # 计算风险等级
        risk_factors = {
            'elevation': data.get('elevation', []),
            'population_density': data.get('population_density', []),
            'infrastructure_vulnerability': data.get('infrastructure_vulnerability', []),
            'historical_disasters': data.get('historical_disasters', [])
        }
        
        # 使用多指标评估方法计算风险
        risk_levels = calculate_risk_levels(risk_factors)
        
        return jsonify({
            'status': 'success',
            'risk_levels': risk_levels
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def calculate_risk_levels(factors):
    """计算综合风险等级"""
    # 示例风险计算逻辑
    risk_levels = {
        'overall_risk': 0.75,
        'sub_risks': {
            'flood_risk': 0.8,
            'infrastructure_risk': 0.7,
            'population_risk': 0.75
        },
        'risk_zones': [
            {
                'id': 'zone1',
                'risk_level': 0.8,
                'coordinates': [[39.9042, 116.4074], [39.9142, 116.4174]]
            }
        ]
    }
    return risk_levels

def create_sample_road_network():
    """创建示例道路网络数据"""
    # 创建一个简单的网格状道路网络
    roads = []
    for i in np.arange(39.88, 39.94, 0.01):
        for j in np.arange(116.38, 116.44, 0.01):
            # 水平道路
            road_h = LineString([
                (j, i),
                (j + 0.01, i)
            ])
            roads.append({
                'geometry': road_h,
                'road_type': 'primary',
                'width': 14
            })

            # 垂直道路
            road_v = LineString([
                (j, i),
                (j, i + 0.01)
            ])
            roads.append({
                'geometry': road_v,
                'road_type': 'secondary',
                'width': 7
            })

    return gpd.GeoDataFrame(roads, crs='EPSG:4326')

def create_evacuation_points_data():
    """创建疏散点数据"""
    points = []
    for point in SAMPLE_EVACUATION_POINTS:
        points.append({
            'geometry': Point(point['lng'], point['lat']),
            'capacity': point['capacity'],
            'type': point['facility_type']
        })
    return gpd.GeoDataFrame(points, crs='EPSG:4326')

def create_risk_areas_data():
    """创建风险区域数据"""
    risk_areas = []
    for area in SAMPLE_RISK_AREAS:
        # 转换坐标为(lng, lat)格式
        coords = [[p[1], p[0]] for p in area['coordinates']]
        coords.append(coords[0])  # 闭合多边形
        risk_areas.append({
            'geometry': LineString(coords),
            'risk_level': area['risk_level']
        })
    return gpd.GeoDataFrame(risk_areas, crs='EPSG:4326') 