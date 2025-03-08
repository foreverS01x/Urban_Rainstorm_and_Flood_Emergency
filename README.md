# 城市洪涝风险分析与应急决策系统

## 项目概述

本系统是一个综合性的城市洪涝灾害风险分析与应急决策支持平台，集成了多源数据分析、智能体模拟、大语言模型决策支持和应急疏散规划等功能。系统主要面向城市防灾减灾部门，为洪涝灾害的预警、响应和疏散提供决策支持。

## 主要功能

### 1. 风险评估模块
- 多源数据整合与分析
- 城市脆弱性评估
- 风险等级计算与分区
- GIS可视化展示

### 2. 智能体模拟模块
- 基于Mesa框架的多智能体模拟
- 市民行为模式建模
- 救援队伍调度模拟
- 疏散过程动态展示

### 3. LLM应急决策支持
- 基于OpenAI API的智能决策
- 多场景应急预案生成
- 实时情况分析与建议
- 资源调配优化建议

### 4. 疏散路径规划
- 多目标疏散路径优化
- 实时路况更新与调整
- 人群分流与容量控制
- 风险感知的路径选择

## 技术架构

### 前端技术
- HTML5/CSS3/JavaScript
- Leaflet地图库
- Bootstrap UI框架
- Font Awesome图标

### 后端技术
- Python Flask Web框架
- GeoPandas空间数据处理
- NetworkX路网分析
- OpenAI API接口
- Mesa智能体框架

### 数据处理
- NumPy/Pandas数据分析
- Scikit-learn机器学习
- Shapely几何计算
- SQLAlchemy数据持久化

## 安装部署

1. 克隆项目
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，设置必要的环境变量（如OpenAI API密钥）
```

5. 运行系统
```bash
python src/web/app.py
```

## 系统配置

### OpenAI API配置
在`.env`文件中设置：
```
OPENAI_API_KEY=your_api_key_here
```

### 地图配置
在`src/web/templates/index.html`中可配置地图中心点和缩放级别：
```javascript
const map = L.map('map').setView([39.9042, 116.4074], 12);
```

### 模拟参数配置
在`src/simulation/config.py`中可调整智能体模拟参数：
```python
SIMULATION_CONFIG = {
    'num_citizens': 100,
    'num_rescue_teams': 5,
    'grid_size': (50, 50),
    'time_steps': 100
}
```

## 数据结构

### 1. 人群聚集点数据
```python
{
    'id': str,
    'lat': float,
    'lng': float,
    'population': int
}
```

### 2. 疏散点数据
```python
{
    'id': str,
    'lat': float,
    'lng': float,
    'facility_type': str,
    'capacity': int,
    'current_occupancy': int
}
```

### 3. 风险区域数据
```python
{
    'coordinates': List[List[float]],
    'risk_level': float
}
```

### 4. 路径规划结果
```python
{
    'coordinates': List[List[float]],
    'evacuation_point_id': str,
    'assigned_population': int,
    'length': float,
    'risk_level': float
}
```

## API接口

### 1. 获取人群聚集点
- 端点：`/api/population_clusters`
- 方法：GET
- 响应：聚集点列表

### 2. 获取疏散点
- 端点：`/api/evacuation_points`
- 方法：GET
- 响应：疏散点列表

### 3. 获取风险区域
- 端点：`/api/risk_areas`
- 方法：GET
- 响应：风险区域列表

### 4. 规划疏散路径
- 端点：`/api/plan_evacuation`
- 方法：POST
- 请求体：
  ```json
  {
    "start_point_id": "string",
    "population": "integer",
    "max_routes": "integer"
  }
  ```

### 5. 更新道路状态
- 端点：`/api/update_road_status`
- 方法：POST
- 请求体：
  ```json
  {
    "road_id": "string",
    "status": "string"
  }
  ```

## 使用说明

1. 风险评估
   - 上传GIS数据
   - 设置评估参数
   - 查看风险分布

2. 智能体模拟
   - 设置模拟参数
   - 启动模拟
   - 观察动态变化

3. 应急决策
   - 输入场景信息
   - 获取LLM建议
   - 查看决策方案

4. 疏散规划
   - 选择起点
   - 设置疏散人数
   - 查看规划路径

## 注意事项

1. 数据安全
   - 定期备份数据
   - 保护敏感信息
   - 控制访问权限

2. 系统维护
   - 定期更新依赖
   - 监控系统性能
   - 记录系统日志

3. 使用限制
   - API调用频率限制
   - 数据大小限制
   - 并发访问限制

## 开发团队

- 项目负责人：[姓名]
- 开发人员：[团队成员]
- 技术支持：[支持团队]

## 版本历史

- v1.0.0 (2024-03)
  - 初始版本发布
  - 基础功能实现

## 许可证

[许可证类型]

## 联系方式

- 邮箱：[电子邮件]
- 网站：[网站地址]
- 电话：[联系电话] 