import pandas as pd
import numpy as np
import os

def generate_discount_factors():
    rows, cols = 8, 10
    actions = 4  # up, down, left, right

    data = []
    for i in range(rows):
        for j in range(cols):
            factors = [f"{np.round(np.random.uniform(0,1),3)}" for _ in range(actions)]
            data.append({
                "Grid 座標": f"{i}_{j}",
                "discount factor": "/".join(factors)
            })

    df = pd.DataFrame(data)
    df.to_csv("hw/hw6/demo code/discount_factor.csv", index=False)
    print("Discount factor file created successfully!")

if __name__ == "__main__":
    generate_discount_factors()
