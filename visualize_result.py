import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

def visualize_results():
    plt.figure(figsize=(10, 6)) #canvas 생성, size는 인치 단위.

    limit_index = 50

    data_sbeed = np.load("results/sbeed_CartPole-v1.npz")
    data_gladius = np.load("results/Gladius_CartPole-v1.npz")
    data_DQN = np.load("results/DQN_CartPole-v1.npz")
    
    plt.plot(data_sbeed['steps'][:limit_index], data_sbeed['returns'][:limit_index], label='SBEED (mean)', color='green', linewidth=2)
    plt.plot(data_gladius['steps'][:limit_index], data_gladius['returns'][:limit_index], label='Gladius (mean)', color='red', linewidth=2)
    plt.plot(data_DQN['steps'][:limit_index], data_DQN['returns'][:limit_index], label='DQN (mean)', color='blue', linewidth=2)


    plt.title(f"Performance")
    plt.xlabel("Gradient Updates")
    plt.ylabel("Average Episode Reward")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() # 여백 정리


    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") #년월일시분초
    save_name = f"{save_dir}/Compare_CartPole-v1.png"

    plt.savefig(save_name, dpi=300)
    print(f"[Vis] Learning curve saved to '{save_name}'")

    plt.show() #show이후에는 canvas를 비워버리므로 savefig를 먼저 해야한다.

visualize_results()