from task import *
import copy

num_schedule = 1  # 调度序列的个数
num_task = 100  # 任务个数
num_task_info = 6  # 任务属性个数
iterx, iterxMax = 0, 100  # 初始迭代次数、最大迭代次数100
if __name__ == '__main__':
    Current_solution = Platform(num_schedule, num_task, num_task_info)
    Current_solution.produce_schedule()
    Current_solution.produce_tasks(num_task, num_task_info)

    init_solution(Current_solution)  # 初始化~
    New_solution = copy.deepcopy(Current_solution)
    Best_solution = copy.deepcopy(Current_solution)
    print(Current_solution.target)
    print(Current_solution.solution)
    print("profits:{}".format(profits(Best_solution)))
    while iterx < iterxMax:  # 终止条件：达到迭代次数，不满足终止条件就缓慢降低温度继续搜索
        while T > 10:
            destroy_index = select_destroy(New_solution)
            repair_index = select_repair(New_solution)
            Current_solution, New_solution, Best_solution = update(Current_solution, New_solution, Best_solution,
                                                                   destroy_index, repair_index)
            print("C_profits:{}".format(profits(Current_solution)))
            print("N_profits:{}".format(profits(New_solution)))
            print("Best_profits:{}".format(profits(Best_solution)))
            print(destroy_score)
            print(repair_score)
            print(wDestroy)
            print(wRepair)
            T = Pa * T  # 温度指数下降
        iterx += 1  # 完成一次降温过程算一次迭代
        T = 100  # 完成一次降温过程算一次迭代
