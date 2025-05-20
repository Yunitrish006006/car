import numpy as np
import os

def convert_qtable_to_txt(npy_file, output_txt_file):
    """將 NumPy 格式的 Q-table 轉換為易讀的文字格式"""
    # 載入 Q-table
    q_table = np.load(npy_file)
    
    # 取得 Q-table 的維度
    rows, cols, actions = q_table.shape
    
    # 開啟輸出文件
    with open(output_txt_file, 'w') as f:
        f.write(f"Q-table shape: {rows}x{cols}x{actions}\n")
        f.write("Format: [Up, Down, Left, Right, UpLeft, UpRight, DownLeft, DownRight]\n\n")
        
        # 遍歷每個狀態
        for i in range(rows):
            for j in range(cols):
                f.write(f"Position ({i},{j}):\n")
                
                # 顯示每個動作的 Q 值
                action_values = []
                for a in range(actions):
                    action_values.append(f"{q_table[i,j,a]:.3f}")
                
                f.write(f"  Q-values: {action_values}\n")
                
                # 找出最佳動作
                best_action = np.argmax(q_table[i,j,:])
                direction = {
                    0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right',
                    4: 'UpLeft', 5: 'UpRight', 6: 'DownLeft', 7: 'DownRight'
                }
                f.write(f"  Best action: {direction.get(best_action, str(best_action))}\n\n")
                
    print(f"Q-table successfully converted to {output_txt_file}")

if __name__ == "__main__":
    input_file = "q_table.npy"
    output_file = "q_table.txt"
    
    convert_qtable_to_txt(input_file, output_file)
