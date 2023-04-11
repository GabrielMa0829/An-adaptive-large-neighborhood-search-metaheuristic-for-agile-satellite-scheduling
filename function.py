from task import *

if __name__ == "__main__":
    Current_solution = Platform(num_schedule, num_task, num_task_info)
    Current_solution.produce_schedule()
    Current_solution.produce_tasks(num_task, num_task_info)

    init_solution(Current_solution)  # 初始化~
    Current_solution.solution = [18, 22, 38, 23, 13, 8, 40, 10, 9, 46, 43, 3, 27, 31, 36, 25, 17, 12, 15, 21, 47, 2, 26]
    print(profits(Current_solution))