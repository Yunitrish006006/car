import matplotlib.pyplot as plt
import numpy as np

# Data for TTL variation
ttl_values = [5, 10, 15, 20, 25]
ttl_delivery_ratio = [52.518, 46.0432, 46.0432, 41.7266, 44.6043]
ttl_overhead = [(146154-73)/(73), (161196-64)/(64), (175691-64)/(64), 
                (167782-58)/(58), (168724-62)/(62)]
ttl_delay = [32.683, 23.3195, 23.2717, 18.406, 12.66]

# Data for numNodes variation
num_nodes = [50, 100, 150, 200, 250]
nodes_delivery_ratio = [0, 20.1439, 46.0432, 17.9856, 100]
nodes_overhead = [float('inf') if d == 0 else (s-d)/d for s, d in 
                 zip([54533, 88467, 161196, 224018, 333445], 
                     [0, 28, 64, 25, 139])]
nodes_delay = [0, 72.7906, 23.3195, 11.5502, 2.00868]

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

# Create second figure for NumNodes variation
plt.figure(figsize=(15, 5))
plt.suptitle('Number of Nodes Variation Analysis', fontsize=16)

# Plot 1: NumNodes Delivery Ratio
plt.subplot(131)
plt.plot(num_nodes, nodes_delivery_ratio, 'r-s')
plt.xlabel('Number of Nodes')
plt.ylabel('Delivery Ratio (%)')
plt.title('Delivery Ratio')
plt.grid(True)

# Plot 2: NumNodes Overhead
plt.subplot(132)
plt.plot(num_nodes[1:], nodes_overhead[1:], 'r-s')  # Skip first point due to infinity
plt.xlabel('Number of Nodes')
plt.ylabel('Overhead Ratio')
plt.title('Overhead')
plt.grid(True)

# Plot 3: NumNodes End-to-End Delay
plt.subplot(133)
plt.plot(num_nodes, nodes_delay, 'r-s')
plt.xlabel('Number of Nodes')
plt.ylabel('End-to-End Delay (s)')
plt.title('End-to-End Delay')
plt.grid(True)

plt.tight_layout()
plt.savefig('nodes_metrics.png')
plt.show()