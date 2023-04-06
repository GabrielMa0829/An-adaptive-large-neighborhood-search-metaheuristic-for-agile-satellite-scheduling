from task import *
import copy

num_schedule = 1  # 调度序列的个数
num_task = 100  # 任务个数
num_task_info = 6  # 任务属性个数

if __name__ == '__main__':
    Current_solution = Platform(num_schedule, num_task, num_task_info)
    Current_solution.produce_schedule()
    Current_solution.produce_tasks(num_task, num_task_info)

    # print(a.target)
    # print(a.tasks)
    # print(a.solution)
    init_solution(Current_solution)
    New_solution = Best_solution = copy.deepcopy(Current_solution)
    print(Current_solution.target)
    print(Current_solution.solution)
    print("profits:{}".format(profits(Current_solution)))
    # destroy_random(New_solution)
    print(New_solution.solution)
    # repair_min_conflict(New_solution)
    # print("profits:{}".format(profits(New_solution)))
    destroy_index = select_destroy(New_solution)
    print(New_solution.solution)
    print("profits:{}".format(profits(New_solution)))
    print(destroy_index)
    repair_index = select_repair(New_solution)
    print(New_solution.solution)
    print("profits:{}".format(profits(New_solution)))
    print(repair_index)
