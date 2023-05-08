from task import *
import copy
import numpy as np
import os


def start(Current_solution: Platform):
    c_iterx = 0
    temp = 100
    init_solution(Current_solution)  # 初始化~
    New_solution = copy.deepcopy(Current_solution)
    Best_solution = copy.deepcopy(Current_solution)
    # print(Current_solution.target)
    # print(Current_solution.solution)
    # print("profits:{}".format(profits(Best_solution)))
    while c_iterx < iterxMax:  # 终止条件：达到迭代次数，不满足终止条件就缓慢降低温度继续搜索
        while temp > 10:
            destroy_index = select_destroy(New_solution)
            repair_index = select_repair(New_solution)
            Current_solution, New_solution, Best_solution = update(Current_solution, New_solution, Best_solution,
                                                                   destroy_index, repair_index)
            # print("C_profits:{}".format(profits(Current_solution)))
            # print("N_profits:{}".format(profits(New_solution)))
            # print("Best_profits:{}".format(profits(Best_solution)))
            # print(destroy_score)
            # print(repair_score)
            # print(wDestroy)
            # print(wRepair)
            temp = Pa * temp  # 温度指数下降
        iterx_prof.append(profits(Current_solution))
        Best_iterx_prof.append(profits(Best_solution))
        c_iterx += 1  # 完成一次降温过程算一次迭代
        temp = 100  # 完成一次降温过程算一次迭代
    print('最优解的利润:{}'.format(profits(Best_solution)))
    return profits(Best_solution), Best_solution.solution, Best_solution.target


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'w')
    for item in range(len(data)):
        s = str(data[item]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    # print("{}保存成功".format(filename))


def save_solution(path, solu: list):
    sol_path = path + '\\solution.txt'
    text_save(sol_path, solu)


path_1 = '.\\data\\EXP_50_1200_1'  # 总的实验数据的保存位置  任务数量_时间窗口长度_序号
exp_num = 50  # 种子数
repeat = 10  # 每个种子重复次数
profit = [[0. for _ in range(repeat + 1)] for _ in range(exp_num)]
if __name__ == '__main__':
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    for i in range(exp_num):  # 执行50组不同的任务序列  种子 100~149
        random_seed = 100 + i

        path_2 = '{}\\{}'.format(path_1, random_seed)
        if not os.path.exists(path_2):
            os.makedirs(path_2)

        solution_list = []
        task_num = []
        for j in range(repeat):  # 每个种子 重复执行 repeat 次
            print("{}-{}".format(i, j))
            solution = Platform(num_schedule, num_task, num_task_info, random_seed)
            solution.produce_schedule()
            solution.produce_tasks(num_task, num_task_info)
            pro, sol, target = start(solution)
            task_num.append(len(sol))
            profit[i][j] = pro
            solution_list.append(sol)
            if not os.path.exists(path_2 + '\\available_time'):
                os.makedirs(path_2 + '\\available_time')
            text_save(path_2 + '\\available_time\\available_time_{}.txt'.format(j), target)
            print("结束后的权重{}".format(wDestroy))
            del solution
            # 重置权重
            reset_weight()
        task_num.append(sum(task_num)/repeat)
        text_save(path_2 + '\\task_num.txt', task_num)  # 在文件中保存任务数量
        save_solution(path_2, solution_list)  # 在文件中保存 解（任务序列）
    for i in range(len(profit)):
        profit[i][-1] = sum(profit[i]) / repeat
    # print(profit)
    text_save(path_1 + '\\profit_data.txt', profit)

    # init_solution(Current_solution)  # 初始化~
    # New_solution = copy.deepcopy(Current_solution)
    # Best_solution = copy.deepcopy(Current_solution)
    # print(Current_solution.target)
    # print(Current_solution.solution)
    # print("profits:{}".format(profits(Best_solution)))
    # while iterx < iterxMax:  # 终止条件：达到迭代次数，不满足终止条件就缓慢降低温度继续搜索
    #     while T > 10:
    #         destroy_index = select_destroy(New_solution)
    #         repair_index = select_repair(New_solution)
    #         Current_solution, New_solution, Best_solution = update(Current_solution, New_solution, Best_solution,
    #                                                                destroy_index, repair_index)
    #         print("C_profits:{}".format(profits(Current_solution)))
    #         print("N_profits:{}".format(profits(New_solution)))
    #         print("Best_profits:{}".format(profits(Best_solution)))
    #         print(destroy_score)
    #         print(repair_score)
    #         print(wDestroy)
    #         print(wRepair)
    #         T = Pa * T  # 温度指数下降
    #     iterx_prof.append(profits(Current_solution))
    #     Best_iterx_prof.append(profits(Best_solution))
    #     iterx += 1  # 完成一次降温过程算一次迭代
    #     T = 100  # 完成一次降温过程算一次迭代
    # print("最优解为：{}".format(Best_solution.solution))
    # print("任务数：{}   利润为：{}    平均利润：{}".format(len(Best_solution.solution), profits(Best_solution),
    #                                                     profits(Best_solution) / len(Best_solution.solution)))
    # print("最优解的时间窗口：{}".format(Best_solution.target))
    # print("最优解的候选任务：(未安排的任务){}".format(Best_solution.list_f))
    # print("性价比：{}".format(performance(Best_solution, Best_solution.solution)))
    # picture_profit()
