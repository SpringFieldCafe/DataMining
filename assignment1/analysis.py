import csv
from datetime import datetime
import os

# 读取文件
def read_files():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建文件的绝对路径
    file1 = os.path.join(script_dir, 'BCW7826_2025-09-25.txt')
    file2 = os.path.join(script_dir, 'BCW7826_2025-09-26.txt')
    
    data = []
    
    # 读取第一个文件
    with open(file1, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    
    # 读取第二个文件
    with open(file2, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    
    return data

# 处理数据并识别状态
def process_data(data):
    processed_data = []
    
    for row in data:
        # 确保行有足够的列
        if len(row) >= 17:
            # 提取时间戳并转换
            start_timestamp = int(row[3])
            end_timestamp = int(row[4])
            
            start_time = datetime.fromtimestamp(start_timestamp/1000)
            end_time = datetime.fromtimestamp(end_timestamp/1000)
            
            # 计算持续时间（秒）
            duration = (end_timestamp - start_timestamp) / 1000
            
            # 提取经纬度
            start_lon = float(row[13])
            start_lat = float(row[14])
            end_lon = float(row[15])
            end_lat = float(row[16])
            
            # 提取其他可能有用的信息
            distance = float(row[5])  # 里程
            avg_speed = float(row[7])  # 平均速度
            max_speed = float(row[8])  # 最大速度
            status_flag = int(row[10])  # 状态标识
            
            # 识别状态
            # 基于数据特征和规则识别状态
            status = identify_status(status_flag, avg_speed, duration, distance)
            
            processed_data.append({
                '时间': row[0],
                '车牌号': row[1],
                '开始时间戳': start_timestamp,
                '结束时间戳': end_timestamp,
                '开始时间': start_time,
                '结束时间': end_time,
                '持续时间': duration,
                '起点经度': start_lon,
                '起点纬度': start_lat,
                '终点经度': end_lon,
                '终点纬度': end_lat,
                '里程': distance,
                '平均速度': avg_speed,
                '最大速度': max_speed,
                '状态标识': status_flag,
                '状态': status
            })
    
    # 按时间排序
    processed_data.sort(key=lambda x: x['开始时间戳'])
    
    return processed_data

# 识别车辆状态
def identify_status(status_flag, avg_speed, duration, distance):
    # 基于规则识别状态
    # 1. 充电状态：速度为0且持续时间较长
    if avg_speed == 0 and duration > 600:  # 10分钟以上静止
        return 'recharging'
    # 2. 载客状态：有状态标识且速度合理
    elif status_flag == 1 or (avg_speed > 0 and distance > 0):
        return 'occupied'
    # 3. 前往接客状态：速度较低且持续时间较短
    elif 0 < avg_speed < 20 and duration < 600:
        return 'heading'
    # 4. 空驶巡游状态：速度较高
    else:
        return 'cruising'

# 分析状态切换
def analyze_status_transitions(data):
    transitions = {}
    previous_status = None
    
    for item in data:
        current_status = item['状态']
        if previous_status and previous_status != current_status:
            key = f'{previous_status}→{current_status}'
            if key in transitions:
                transitions[key] += 1
            else:
                transitions[key] = 1
        previous_status = current_status
    
    return transitions

# 计算各状态持续时间
def calculate_status_durations(data):
    durations = {
        'occupied': 0,
        'heading': 0,
        'recharging': 0,
        'cruising': 0
    }
    
    for item in data:
        status = item['状态']
        if status in durations:
            durations[status] += item['持续时间']
    
    return durations

# 创建HTML可视化文件
def create_html_visualization(data):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'BCW7826驾驶行为分析.html')
    
    # 分析状态切换
    transitions = analyze_status_transitions(data)
    
    # 计算各状态持续时间
    durations = calculate_status_durations(data)
    total_duration = sum(durations.values())
    
    # 准备数据JS
    data_js = '['
    for item in data:
        data_js += '{ ' 
        data_js += f'start_lon: {item["起点经度"]}, ' 
        data_js += f'start_lat: {item["起点纬度"]}, ' 
        data_js += f'end_lon: {item["终点经度"]}, ' 
        data_js += f'end_lat: {item["终点纬度"]}, ' 
        data_js += f'start_time: "{item["开始时间"].strftime("%Y-%m-%d %H:%M:%S")}", ' 
        data_js += f'end_time: "{item["结束时间"].strftime("%Y-%m-%d %H:%M:%S")}", ' 
        data_js += f'duration: {item["持续时间"]}, ' 
        data_js += f'status: "{item["状态"]}", ' 
        data_js += f'avg_speed: {item["平均速度"]}, ' 
        data_js += f'distance: {item["里程"]} ' 
        data_js += '},'
    data_js = data_js.rstrip(',') + ']'
    
    # 准备状态切换数据
    transitions_js = '['
    for key, value in transitions.items():
        from_status, to_status = key.split('→')
        transitions_js += f'{{ from: "{from_status}", to: "{to_status}", value: {value} }},'
    transitions_js = transitions_js.rstrip(',') + ']'
    
    # 准备状态持续时间数据
    durations_js = '['
    for status, duration in durations.items():
        percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
        durations_js += f'{{ name: "{status}", value: {duration}, percentage: {percentage:.2f} }},'
    durations_js = durations_js.rstrip(',') + ']'
    
    # 构建HTML内容
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>BCW7826 驾驶行为分析</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 引入Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <!-- 引入Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <!-- 引入ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .section {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .section h2 {
            margin-top: 0;
            color: #555;
        }
        #map {
            height: 500px;
            width: 100%;
            border-radius: 4px;
        }
        .chart-container {
            height: 400px;
            margin: 20px 0;
        }
        .status-filter {
            margin: 10px 0;
        }
        .status-filter label {
            margin-right: 15px;
            cursor: pointer;
        }
        .timeline {
            position: relative;
            height: 100px;
            margin: 20px 0;
            border-left: 2px solid #333;
            padding-left: 20px;
        }
        .timeline-item {
            position: relative;
            margin-bottom: 10px;
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -25px;
            top: 5px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }
        .timeline-item.occupied::before {
            background-color: #e74c3c;
        }
        .timeline-item.heading::before {
            background-color: #f39c12;
        }
        .timeline-item.recharging::before {
            background-color: #27ae60;
        }
        .timeline-item.cruising::before {
            background-color: #3498db;
        }
        .summary {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }
        .summary h3 {
            margin-top: 0;
        }
        .status-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 10px 0;
        }
        .status-legend-item {
            display: flex;
            align-items: center;
        }
        .status-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 5px;
        }
        .status-color.occupied {
            background-color: #e74c3c;
        }
        .status-color.heading {
            background-color: #f39c12;
        }
        .status-color.recharging {
            background-color: #27ae60;
        }
        .status-color.cruising {
            background-color: #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>粤BCW7826 驾驶行为分析</h1>
        
        <!-- 状态筛选 -->
        <div class="section">
            <h2>状态筛选</h2>
            <div class="status-filter">
                <label><input type="checkbox" class="status-checkbox" value="occupied" checked> 载客 (occupied)</label>
                <label><input type="checkbox" class="status-checkbox" value="heading" checked> 前往接客 (heading)</label>
                <label><input type="checkbox" class="status-checkbox" value="recharging" checked> 充电 (recharging)</label>
                <label><input type="checkbox" class="status-checkbox" value="cruising" checked> 空驶巡游 (cruising)</label>
            </div>
            <div class="status-legend">
                <div class="status-legend-item">
                    <div class="status-color occupied"></div>
                    <span>载客</span>
                </div>
                <div class="status-legend-item">
                    <div class="status-color heading"></div>
                    <span>前往接客</span>
                </div>
                <div class="status-legend-item">
                    <div class="status-color recharging"></div>
                    <span>充电</span>
                </div>
                <div class="status-legend-item">
                    <div class="status-color cruising"></div>
                    <span>空驶巡游</span>
                </div>
            </div>
        </div>
        
        <!-- 地图轨迹 -->
        <div class="section">
            <h2>地理轨迹展示</h2>
            <div id="map"></div>
        </div>
        
        <!-- 时间轴 -->
        <div class="section">
            <h2>时间轴可视化</h2>
            <div id="timeline" class="timeline"></div>
        </div>
        
        <!-- 统计图表 -->
        <div class="section">
            <h2>数据统计</h2>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div class="chart-container" style="flex: 1; min-width: 400px;">
                    <h3>状态持续时间占比</h3>
                    <div id="durationChart"></div>
                </div>
                <div class="chart-container" style="flex: 1; min-width: 400px;">
                    <h3>状态切换次数</h3>
                    <div id="transitionChart"></div>
                </div>
            </div>
        </div>
        
        <!-- 状态转移图 -->
        <div class="section">
            <h2>状态转移图</h2>
            <div class="chart-container">
                <div id="stateTransitionChart"></div>
            </div>
        </div>
        
        <!-- 数据分析结论 -->
        <div class="section">
            <h2>数据分析结论</h2>
            <div class="summary">
                <h3>驾驶行为分析总结</h3>
                <p><strong>总行驶时间：</strong><span id="totalTime"></span></p>
                <p><strong>总行驶里程：</strong><span id="totalDistance"></span></p>
                <p><strong>各状态占比：</strong></p>
                <ul id="statusPercentage"></ul>
                <p><strong>主要状态转换：</strong></p>
                <ul id="mainTransitions"></ul>
                <p><strong>驾驶行为特点：</strong></p>
                <p id="drivingCharacteristics"></p>
            </div>
        </div>
    </div>
    
    <script>
        // 数据
        var data = {data_js};
        var transitions = {transitions_js};
        var durations = {durations_js};
        
        // 状态颜色映射
        var statusColors = {
            'occupied': '#e74c3c',    // 红色
            'heading': '#f39c12',     // 橙色
            'recharging': '#27ae60',   // 绿色
            'cruising': '#3498db'      // 蓝色
        };
        
        // 初始化地图
        var map = L.map('map').setView([22.55, 114.1], 12);
        
        // 添加底图
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        
        // 绘制轨迹
        var routeLayers = [];
        var markerLayers = [];
        
        function drawRoutes() {
            // 清除现有轨迹
            routeLayers.forEach(function(layer) {
                map.removeLayer(layer);
            });
            markerLayers.forEach(function(layer) {
                map.removeLayer(layer);
            });
            routeLayers = [];
            markerLayers = [];
            
            // 获取选中的状态
            var selectedStatuses = [];
            document.querySelectorAll('.status-checkbox:checked').forEach(function(checkbox) {
                selectedStatuses.push(checkbox.value);
            });
            
            // 绘制轨迹
            data.forEach(function(item) {
                if (selectedStatuses.includes(item.status)) {
                    // 绘制路线
                    var latlngs = [
                        [item.start_lat, item.start_lon],
                        [item.end_lat, item.end_lon]
                    ];
                    
                    var polyline = L.polyline(latlngs, {
                        color: statusColors[item.status],
                        weight: 3,
                        opacity: 0.7
                    }).addTo(map);
                    routeLayers.push(polyline);
                    
                    // 添加起点标记
                    var startMarker = L.marker([item.start_lat, item.start_lon], {
                        icon: L.divIcon({
                            className: 'custom-div-icon',
                            html: '<div style="background-color:' + statusColors[item.status] + '; width:8px; height:8px; border-radius:50%;"></div>',
                            iconSize: [8, 8],
                            iconAnchor: [4, 4]
                        })
                    }).addTo(map);
                    markerLayers.push(startMarker);
                    
                    // 添加弹出信息
                    startMarker.bindPopup(
                        '<b>起点</b><br/>' +
                        '时间: ' + item.start_time + '<br/>' +
                        '状态: ' + item.status + '<br/>' +
                        '平均速度: ' + item.avg_speed.toFixed(2) + ' km/h<br/>' +
                        '里程: ' + item.distance.toFixed(2) + ' km<br/>' +
                        '坐标: (' + item.start_lon.toFixed(6) + ', ' + item.start_lat.toFixed(6) + ')'
                    );
                }
            });
        }
        
        // 绘制时间轴
        function drawTimeline() {
            var timeline = document.getElementById('timeline');
            timeline.innerHTML = '';
            
            // 获取选中的状态
            var selectedStatuses = [];
            document.querySelectorAll('.status-checkbox:checked').forEach(function(checkbox) {
                selectedStatuses.push(checkbox.value);
            });
            
            // 绘制时间轴项目
            data.forEach(function(item, index) {
                if (selectedStatuses.includes(item.status)) {
                    var itemElement = document.createElement('div');
                    itemElement.className = 'timeline-item ' + item.status;
                    
                    var startTime = new Date(item.start_time);
                    var hours = startTime.getHours();
                    var minutes = startTime.getMinutes();
                    var timeStr = hours.toString().padStart(2, '0') + ':' + minutes.toString().padStart(2, '0');
                    
                    itemElement.innerHTML = '<strong>' + timeStr + '</strong> - ' + item.status + ' (持续 ' + Math.round(item.duration / 60) + ' 分钟)';
                    timeline.appendChild(itemElement);
                }
            });
        }
        
        // 绘制状态持续时间饼图
        function drawDurationChart() {
            var chart = echarts.init(document.getElementById('durationChart'));
            
            var option = {
                tooltip: {
                    trigger: 'item',
                    formatter: '{a} <br/>{b}: {c}分钟 ({d}%)'
                },
                legend: {
                    orient: 'vertical',
                    left: 'left',
                    data: durations.map(function(item) {
                        return item.name;
                    })
                },
                series: [
                    {
                        name: '状态持续时间',
                        type: 'pie',
                        radius: '50%',
                        data: durations.map(function(item) {
                            return {
                                value: Math.round(item.value / 60), // 转换为分钟
                                name: item.name
                            };
                        }),
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 10,
                                shadowOffsetX: 0,
                                shadowColor: 'rgba(0, 0, 0, 0.5)'
                            }
                        },
                        itemStyle: {
                            color: function(params) {
                                return statusColors[params.name];
                            }
                        }
                    }
                ]
            };
            
            chart.setOption(option);
        }
        
        // 绘制状态切换次数柱状图
        function drawTransitionChart() {
            var chart = echarts.init(document.getElementById('transitionChart'));
            
            var option = {
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'shadow'
                    }
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: transitions.map(function(item) {
                        return item.from + '→' + item.to;
                    }),
                    axisLabel: {
                        rotate: 45
                    }
                },
                yAxis: {
                    type: 'value',
                    name: '切换次数'
                },
                series: [
                    {
                        name: '切换次数',
                        type: 'bar',
                        data: transitions.map(function(item) {
                            return item.value;
                        }),
                        itemStyle: {
                            color: function(params) {
                                var fromStatus = params.name.split('→')[0];
                                return statusColors[fromStatus];
                            }
                        }
                    }
                ]
            };
            
            chart.setOption(option);
        }
        
        // 绘制状态转移图
        function drawStateTransitionChart() {
            var chart = echarts.init(document.getElementById('stateTransitionChart'));
            
            var nodes = [
                { name: 'occupied', symbolSize: 80, itemStyle: { color: statusColors.occupied } },
                { name: 'heading', symbolSize: 80, itemStyle: { color: statusColors.heading } },
                { name: 'recharging', symbolSize: 80, itemStyle: { color: statusColors.recharging } },
                { name: 'cruising', symbolSize: 80, itemStyle: { color: statusColors.cruising } }
            ];
            
            var links = transitions.map(function(item) {
                return {
                    source: item.from,
                    target: item.to,
                    value: item.value
                };
            });
            
            var option = {
                tooltip: {
                    trigger: 'item',
                    formatter: function(params) {
                        if (params.dataType === 'edge') {
                            return params.data.source + ' → ' + params.data.target + '<br/>次数: ' + params.data.value;
                        } else {
                            return params.name;
                        }
                    }
                },
                series: [
                    {
                        type: 'graph',
                        layout: 'force',
                        data: nodes,
                        links: links,
                        roam: true,
                        label: {
                            show: true,
                            position: 'inside',
                            color: '#fff',
                            fontSize: 12
                        },
                        lineStyle: {
                            color: 'source',
                            curveness: 0.3
                        },
                        emphasis: {
                            focus: 'adjacency',
                            lineStyle: {
                                width: 4
                            }
                        }
                    }
                ]
            };
            
            chart.setOption(option);
        }
        
        // 更新数据分析结论
        function updateAnalysisSummary() {
            // 计算总时间
            var totalSeconds = durations.reduce(function(sum, item) {
                return sum + item.value;
            }, 0);
            var totalHours = totalSeconds / 3600;
            document.getElementById('totalTime').textContent = totalHours.toFixed(2) + ' 小时';
            
            // 计算总里程
            var totalDistance = data.reduce(function(sum, item) {
                return sum + item.distance;
            }, 0);
            document.getElementById('totalDistance').textContent = totalDistance.toFixed(2) + ' 公里';
            
            // 更新状态占比
            var statusPercentage = document.getElementById('statusPercentage');
            statusPercentage.innerHTML = '';
            durations.forEach(function(item) {
                var li = document.createElement('li');
                li.innerHTML = item.name + ': ' + item.percentage.toFixed(1) + '%';
                statusPercentage.appendChild(li);
            });
            
            // 更新主要状态转换
            var mainTransitions = document.getElementById('mainTransitions');
            mainTransitions.innerHTML = '';
            transitions.sort(function(a, b) {
                return b.value - a.value;
            }).slice(0, 5).forEach(function(item) {
                var li = document.createElement('li');
                li.innerHTML = item.from + ' → ' + item.to + ': ' + item.value + ' 次';
                mainTransitions.appendChild(li);
            });
            
            // 更新驾驶行为特点
            var drivingCharacteristics = document.getElementById('drivingCharacteristics');
            var characteristics = '';
            
            // 分析主要状态
            var mainStatus = durations.reduce(function(max, item) {
                return item.value > max.value ? item : max;
            }, durations[0]);
            characteristics += '主要状态为 ' + mainStatus.name + '，占比 ' + mainStatus.percentage.toFixed(1) + '%。';
            
            // 分析平均速度
            var avgSpeed = data.reduce(function(sum, item) {
                return sum + item.avg_speed;
            }, 0) / data.length;
            characteristics += ' 平均速度为 ' + avgSpeed.toFixed(2) + ' km/h。';
            
            drivingCharacteristics.textContent = characteristics;
        }
        
        // 初始化
        function init() {
            drawRoutes();
            drawTimeline();
            drawDurationChart();
            drawTransitionChart();
            drawStateTransitionChart();
            updateAnalysisSummary();
        }
        
        // 状态筛选事件
        document.querySelectorAll('.status-checkbox').forEach(function(checkbox) {
            checkbox.addEventListener('change', function() {
                drawRoutes();
                drawTimeline();
            });
        });
        
        // 窗口大小变化时重新调整图表
        window.addEventListener('resize', function() {
            var charts = ['durationChart', 'transitionChart', 'stateTransitionChart'];
            charts.forEach(function(chartId) {
                var chart = echarts.getInstanceByDom(document.getElementById(chartId));
                if (chart) {
                    chart.resize();
                }
            });
        });
        
        // 初始化
        init();
    </script>
</body>
</html>
'''
    
    # 替换数据
    html_content = html_content.replace('{data_js}', data_js)
    html_content = html_content.replace('{transitions_js}', transitions_js)
    html_content = html_content.replace('{durations_js}', durations_js)
    
    # 写入HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'HTML可视化文件已保存到 {output_file}')
    print('请在浏览器中打开此文件查看驾驶行为分析')

# 保存处理后的数据
def save_processed_data(data, output_file='processed_data.csv'):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建输出文件的绝对路径
    output_file = os.path.join(script_dir, output_file)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['时间', '车牌号', '开始时间戳', '结束时间戳', '开始时间', '结束时间', '持续时间', '起点经度', '起点纬度', '终点经度', '终点纬度', '里程', '平均速度', '最大速度', '状态标识', '状态']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in data:
            writer.writerow(item)
    print(f'处理后的数据已保存到 {output_file}')

# 主函数
def main():
    # 读取数据
    data = read_files()
    
    # 处理数据
    processed_data = process_data(data)
    
    # 显示数据信息
    print('数据行数:', len(processed_data))
    print('\n前5行数据:')
    for i, item in enumerate(processed_data[:5]):
        print(f'第{i+1}行:')
        print(f'  时间: {item["时间"]}')
        print(f'  车牌号: {item["车牌号"]}')
        print(f'  开始时间: {item["开始时间"]}')
        print(f'  结束时间: {item["结束时间"]}')
        print(f'  持续时间: {item["持续时间"]} 秒')
        print(f'  起点: ({item["起点经度"]}, {item["起点纬度"]})')
        print(f'  终点: ({item["终点经度"]}, {item["终点纬度"]})')
        print(f'  状态: {item["状态"]}')
        print()
    
    # 创建HTML可视化
    create_html_visualization(processed_data)
    
    # 保存处理后的数据
    save_processed_data(processed_data)

if __name__ == '__main__':
    main()
