import copy, math, time
import numpy as np
def GKK_solver(w_list, v_list, N, W, used_flag=None, sorted_id=None):
    """_summary_

    Args:
        w (list): weight of the items
        v (list): value of the items
        N (int): number of the items
        W (int): Total capacity of the package
    """
    used_flag = ([False] * N) if used_flag is None else used_flag

        
    if sorted_id is None:
        GKK_v = copy.deepcopy(v_list)
        for k, value in enumerate(GKK_v):
            GKK_v[k] = value / w_list[k]
        sorted_id = sorted(range(len(GKK_v)), key=lambda k: GKK_v[k], reverse=True)
        GKK_v.sort(reverse=True)
    
        # print(sorted_id, GKK_v)
    
    # phase 1
    ans = 0
    select_flag = [False] * N
    for k in sorted_id:
        if W >= w_list[k] and not used_flag[k]:
            select_flag[k] = True
            ans += v_list[k]; W -= w_list[k]
    
    # phase 2        
    max_w, max_v = max(w_list), max(v_list)
    if max_w <= W and max_v > ans:
        select_flag = [False] * N
        select_flag[v_list.index(max_v)] = True
        ans = max_v

    return ans
    
def ordered_idx_generator(start, end, number):
    if number > end - start + 1:
        return []
    if number == 1:
        return [[i] for i in range(start, end + 1)]
    result = []
    for i in range(start, end + 1):
        sub_results = ordered_idx_generator(i + 1, end, number - 1)
        for sub_result in sub_results:
            result.append([i] + sub_result)
    return result
             
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
             
def approximate_solver(w_list, v_list, N, W, eta=1):
    """_summary_

    Args:
        w (list): weight of the items
        v (list): value of the items
        N (int): number of the items
        W (int): Total capacity of the package
        eta (float): parameter of the approximate algorithm
    """
    
    m = math.ceil(1 / eta)
    GKK_v = copy.deepcopy(v_list)
    for k, value in enumerate(GKK_v):
        GKK_v[k] = value / w_list[k]
        
    max_value = 0
    sorted_id = sorted(range(len(GKK_v)), key=lambda k: GKK_v[k], reverse=True)
    for t in range(1, m + 1):
        ordered_list = ordered_idx_generator(0, N-1, t)
        for ordered_idxs in ordered_list:
            # print(ordered_idxs)
            sum_w, sum_v, used_flag = 0, 0, [False] * N
            
            break_flag = False
            for idx in ordered_idxs:
                if sum_w + w_list[idx] > W: 
                    break_flag = True
                    break
                sum_w += w_list[idx]; sum_v += v_list[idx]
                used_flag[idx] = True      
                     
            if not break_flag:
                sum_v += GKK_solver(w_list, v_list, N, W-sum_w, used_flag, sorted_id)
                max_value = max(max_value, sum_v)
    # exit(0)
    return max_value
    
def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        N, W, T = map(int, lines[0].strip().split())
        w_list, v_list = [], []
        for k, line in enumerate(lines[1:]):
            if k >= N: break
            a, b = map(int, line.strip().split())
            w_list.append(a); v_list.append(b)
    return N, W, T, w_list, v_list
    
def summarize(GKK_rate_list, st, ed, step):
    cond_list = []
    
    for k in np.arange(st, ed, step):
        condition = lambda x: (x >= k and x < k+step)
        count = sum(1 for element in GKK_rate_list if condition(element))
        cond_list.append(count)
        
    condition = lambda x: (x >= ed)
    count = sum(1 for element in GKK_rate_list if condition(element))
    cond_list.append(count)
    
    return cond_list

if __name__ == '__main__':
    # miniN
    GKK_rate_list, epsion_rate_list = [], []
    save_dir = 'miniN'
    for sample in range(500):
        N, W, T, w_list, v_list = read_data_from_file(f'{save_dir}//{sample}.txt')
        # print(N, W, T, w_list, v_list)
        print('---------------------------------')
        print(f'sample {sample}: ', N, W, T)
        # print(exact_solver(w_list, v_list, N, W))
        exact_ans = T
        GKK_approximate_ans = GKK_solver(w_list, v_list, N, W)
        epsion_approximate_ans = approximate_solver(w_list, v_list, N, W)
        
        GKK_rate = exact_ans / GKK_approximate_ans
        epsion_rate = exact_ans / epsion_approximate_ans
        
        GKK_rate_list.append(GKK_rate)
        epsion_rate_list.append(epsion_rate)
        
        print('GKK_rate', '%.6f'%GKK_rate)
        print('epsion_rate', '%.6f'%epsion_rate)
        
    print('---------------------------------')
    print('---------------------------------')
    GKK_rate_mean, GKK_rate_var = np.mean(GKK_rate_list), np.var(GKK_rate_list) 
    epsion_rate_mean, epsion_rate_var = np.mean(epsion_rate_list), np.var(epsion_rate_list) 
    print(GKK_rate_mean, GKK_rate_var)
    print(epsion_rate_mean, epsion_rate_var)
    
    # cond_list = summarize(GKK_rate_list, st=1, ed=1.3, step=0.05)
    cond_list = summarize(epsion_rate_list, st=1, ed=1.2, step=0.05)
    print(cond_list)
    # print(epsion_rate_mean, epsion_rate_var)
    
    
        
    