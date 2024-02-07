import os, random
import numpy as np

def exact_solver(w_list, v_list, N, W):
    dp = np.zeros((N, W+1))
    
    # 动态规划求解
    for i in range(N):
        for j in range(W+1):
            # 如果当前物品的重量大于背包容量，则无法放入背包
            if w_list[i] > j:
                dp[i][j] = dp[i - 1][j] if i >= 1 else 0
            else:
                # 考虑将当前物品放入背包和不放入背包两种情况，取价值最大的方案
                if i >= 1:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w_list[i]] + v_list[i])
                else:
                    dp[i][j] = v_list[i]
    
    return int(dp[N-1][W])
   

def generator(N, M=1000, w_range=(20, 200), v_range=(3, 30)):
    w_list = [random.randint(w_range[0], w_range[1]) for _ in range(N)]
    v_list = [random.randint(v_range[0], v_range[1]) for _ in range(N)]
        
    return N, M, w_list, v_list

if __name__ == '__main__':
    # # mini item number
    # save_dir = 'miniN'
    # os.makedirs(save_dir, exist_ok=True)
    # for idx in range(500):
    #     N, M = 5, 100
    #     N, M, w_list, v_list = generator(N, M, w_range=(30,70), v_range=(30, 70))
    #     best_ans = exact_solver(w_list, v_list, N, M)
        
    #     save_path = f'{save_dir}//{idx}.txt'
    #     assert not os.path.exists(save_path)
        
    #     saved_data = [[str(N), str(M), str(best_ans)]]
        
    #     for k in range(N):
    #         saved_data.append([str(w_list[k]), str(v_list[k])])
        
    #     with open(save_path, 'w') as f:
    #         for row in saved_data:
    #             line = ' '.join(row) + '\n'
    #             f.write(line)
            
    #         print(N, M, best_ans)
            # exit(0)
    
    # huge item number
    save_dir = 'largeN'
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(500):
        N, M = 100, 1000
        N, M, w_list, v_list = generator(N, M, w_range=(30,70), v_range=(30, 70))
        best_ans = exact_solver(w_list, v_list, N, M)
        
        save_path = f'{save_dir}//{idx}.txt'
        assert not os.path.exists(save_path)
        
        saved_data = [[str(N), str(M), str(best_ans)]]
        
        for k in range(N):
            saved_data.append([str(w_list[k]), str(v_list[k])])
        
        with open(save_path, 'w') as f:
            for row in saved_data:
                line = ' '.join(row) + '\n'
                f.write(line)
            
            print(N, M, best_ans)
    
    
    
    