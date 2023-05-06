from task import *
import copy

if __name__ == '__main__':
    Current_solution = Platform(num_schedule, num_task, num_task_info, 105)
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
        iterx_prof.append(profits(Current_solution))
        Best_iterx_prof.append(profits(Best_solution))
        iterx += 1  # 完成一次降温过程算一次迭代
        T = 100  # 完成一次降温过程算一次迭代
    print("最优解为：{}".format(Best_solution.solution))
    print("任务数：{}   利润为：{}    平均利润：{}".format(len(Best_solution.solution), profits(Best_solution),
                                                        profits(Best_solution) / len(Best_solution.solution)))
    print("最优解的时间窗口：{}".format(Best_solution.target))
    print("最优解的候选任务：(未安排的任务){}".format(Best_solution.list_f))
    print("性价比：{}".format(performance(Best_solution, Best_solution.solution)))
    picture_profit()
