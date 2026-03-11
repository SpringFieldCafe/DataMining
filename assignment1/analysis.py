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

# 处理数据
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
            
            # 提取经纬度
            start_lon = float(row[13])
            start_lat = float(row[14])
            end_lon = float(row[15])
            end_lat = float(row[16])
            
            processed_data.append({
                '时间': row[0],
                '车牌号': row[1],
                '开始时间戳': start_timestamp,
                '结束时间戳': end_timestamp,
                '开始时间': start_time,
                '结束时间': end_time,
                '起点经度': start_lon,
                '起点纬度': start_lat,
                '终点经度': end_lon,
                '终点纬度': end_lat
            })
    
    return processed_data

# 创建HTML可视化文件
def create_html_visualization(data):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'BCW7826轨迹可视化.html')
    
    # 准备数据JS
    data_js = '['
    for item in data:
        data_js += '{ ' 
        data_js += f'start_lon: {item["起点经度"]}, ' 
        data_js += f'start_lat: {item["起点纬度"]}, ' 
        data_js += f'end_lon: {item["终点经度"]}, ' 
        data_js += f'end_lat: {item["终点纬度"]}, ' 
        data_js += f'start_time: "{item["开始时间"].strftime("%Y-%m-%d %H:%M:%S")}", ' 
        data_js += f'end_time: "{item["结束时间"].strftime("%Y-%m-%d %H:%M:%S")}" ' 
        data_js += '},'
    data_js = data_js.rstrip(',') + ']'
    
    # 构建HTML内容
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>BCW7826 车辆行驶轨迹可视化</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 引入Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <!-- 引入Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <style>
        #map {
            height: 600px;
            width: 100%;
        }
        .info {
            padding: 6px 8px;
            font: 14px/16px Arial, Helvetica, sans-serif;
            background: white;
            background: rgba(255,255,255,0.8);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            border-radius: 5px;
        }
        .legend {
            line-height: 18px;
            color: #555;
        }
        .legend i {
            width: 18px;
            height: 18px;
            float: left;
            margin-right: 8px;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <h1>BCW7826 车辆行驶轨迹可视化</h1>
    <div id="map"></div>
    <div class="info" style="margin-top: 10px;">
        <h3>轨迹信息</h3>
        <p>总记录数: {total_records}</p>
        <p>开始时间: {start_time}</p>
        <p>结束时间: {end_time}</p>
    </div>
    
    <script>
        // 初始化地图
        var map = L.map('map').setView([22.55, 114.1], 12);
        
        // 添加底图
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        
        // 定义图标
        var startIcon = L.divIcon({
            className: 'custom-div-icon',
            html: '<div style="background-color:blue; width:12px; height:12px; border-radius:50%;"></div>',
            iconSize: [12, 12],
            iconAnchor: [6, 6]
        });
        
        var endIcon = L.divIcon({
            className: 'custom-div-icon',
            html: '<div style="background-color:red; width:12px; height:12px; border-radius:50%;"></div>',
            iconSize: [12, 12],
            iconAnchor: [6, 6]
        });
        
        // 添加轨迹数据
        var data = {data_js};
        
        // 绘制轨迹和标记
        data.forEach(function(item, index) {
            // 绘制路线
            var latlngs = [
                [item.start_lat, item.start_lon],
                [item.end_lat, item.end_lon]
            ];
            
            var polyline = L.polyline(latlngs, {
                color: 'black',
                weight: 2,
                opacity: 0.3
            }).addTo(map);
            
            // 添加起点标记
            var startMarker = L.marker([item.start_lat, item.start_lon], {
                icon: startIcon
            }).addTo(map);
            
            // 添加起点弹出信息
            startMarker.bindPopup(
                '<b>起点</b><br/>' +
                '时间: ' + item.start_time + '<br/>' +
                '坐标: (' + item.start_lon.toFixed(6) + ', ' + item.start_lat.toFixed(6) + ')'
            );
            
            // 添加终点标记
            var endMarker = L.marker([item.end_lat, item.end_lon], {
                icon: endIcon
            }).addTo(map);
            
            // 添加终点弹出信息
            endMarker.bindPopup(
                '<b>终点</b><br/>' +
                '时间: ' + item.end_time + '<br/>' +
                '坐标: (' + item.end_lon.toFixed(6) + ', ' + item.end_lat.toFixed(6) + ')'
            );
        });
        
        // 添加图例
        var legend = L.control({
            position: 'bottomright'
        });
        
        legend.onAdd = function(map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML = '<i style="background:blue"></i> 起点<br/>' +
                           '<i style="background:red"></i> 终点<br/>' +
                           '<i style="background:black"></i> 行驶路线';
            return div;
        };
        
        legend.addTo(map);
    </script>
</body>
</html>
'''
    
    # 替换数据
    html_content = html_content.replace('{total_records}', str(len(data)))
    html_content = html_content.replace('{start_time}', data[0]['开始时间'].strftime('%Y-%m-%d %H:%M:%S'))
    html_content = html_content.replace('{end_time}', data[-1]['结束时间'].strftime('%Y-%m-%d %H:%M:%S'))
    html_content = html_content.replace('{data_js}', data_js)
    
    # 写入HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'HTML可视化文件已保存到 {output_file}')
    print('请在浏览器中打开此文件查看轨迹可视化')

# 保存处理后的数据
def save_processed_data(data, output_file='processed_data.csv'):
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建输出文件的绝对路径
    output_file = os.path.join(script_dir, output_file)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['时间', '车牌号', '开始时间戳', '结束时间戳', '开始时间', '结束时间', '起点经度', '起点纬度', '终点经度', '终点纬度']
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
        print(f'  起点: ({item["起点经度"]}, {item["起点纬度"]})')
        print(f'  终点: ({item["终点经度"]}, {item["终点纬度"]})')
        print()
    
    # 创建HTML可视化
    create_html_visualization(processed_data)
    
    # 保存处理后的数据
    save_processed_data(processed_data)

if __name__ == '__main__':
    main()
