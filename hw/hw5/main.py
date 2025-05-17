import matplotlib.pyplot as plt
import numpy as np
import re

def parse_data_file(filename):
    values = []
    delivery_ratios = []
    packets_sent = []
    packets_delivered = []
    delays = []
    
    with open(filename, 'r') as file:
        current_data = {}
        for line in file:
            if line.startswith('*'):
                values.append(float(line.strip('* \n')))
            elif 'Packets sent:' in line:
                packets_sent.append(int(re.findall(r'\d+', line)[0]))
            elif 'Packets delivered:' in line:
                packets_delivered.append(int(re.findall(r'\d+', line)[0]))
            elif 'Delivery percentage:' in line:
                delivery_ratios.append(float(re.findall(r'[\d.]+', line)[0]))
            elif 'Average End-to-End Delay:' in line:
                delay = float(re.findall(r'[\d.]+', line)[0]) if 'nan' not in line else 0
                delays.append(delay)

    overhead = [float('inf') if d == 0 else (s-d)/d 
               for s, d in zip(packets_sent, packets_delivered)]
    
    return values, delivery_ratios, overhead, delays

# Read data from files
ttl_values, ttl_delivery_ratio, ttl_overhead, ttl_delay = parse_data_file('/home/yunitrish/workspace/car/hw/hw5/ttl.txt')
distance_values, distance_delivery_ratio, distance_overhead, distance_delay = parse_data_file('/home/yunitrish/workspace/car/hw/hw5/distance.txt')

# Create first figure for TTL variation
plt.figure(figsize=(15, 5))
plt.suptitle('TTL Variation Analysis', fontsize=16)

# Plot 1: TTL Delivery Ratio
plt.subplot(131)
plt.plot(ttl_values, ttl_delivery_ratio, 'b-o')
plt.xlabel('TTL')
plt.ylabel('Delivery Ratio (%)')
plt.title('Delivery Ratio')
plt.grid(True)

# Plot 2: TTL Overhead
plt.subplot(132)
plt.plot(ttl_values, ttl_overhead, 'b-o')
plt.xlabel('TTL')
plt.ylabel('Overhead Ratio')
plt.title('Overhead')
plt.grid(True)

# Plot 3: TTL End-to-End Delay
plt.subplot(133)
plt.plot(ttl_values, ttl_delay, 'b-o')
plt.xlabel('TTL')
plt.ylabel('End-to-End Delay (s)')
plt.title('End-to-End Delay')
plt.grid(True)

plt.tight_layout()
plt.savefig('ttl_metrics.png')

# Create second figure for Distance variation
plt.figure(figsize=(15, 5))
plt.suptitle('Distance Variation Analysis', fontsize=16)

# Plot 1: Distance Delivery Ratio
plt.subplot(131)
plt.plot(distance_values, distance_delivery_ratio, 'r-s')
plt.xlabel('Distance')
plt.ylabel('Delivery Ratio (%)')
plt.title('Delivery Ratio')
plt.grid(True)

# Plot 2: Distance Overhead
plt.subplot(132)
plt.plot(distance_values[1:], distance_overhead[1:], 'r-s')  # Skip first point due to infinity
plt.xlabel('Distance')
plt.ylabel('Overhead Ratio')
plt.title('Overhead')
plt.grid(True)

# Plot 3: Distance End-to-End Delay
plt.subplot(133)