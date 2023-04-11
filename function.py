from task import *

if __name__ == "__main__":
    Current_solution = Platform(num_schedule, num_task, num_task_info)
    Current_solution.produce_schedule()
    Current_solution.produce_tasks(num_task, num_task_info)

    init_solution(Current_solution)  # 初始化~
    New_solution = copy.deepcopy(Current_solution)
    Best_solution = copy.deepcopy(Current_solution)
    print(Current_solution.tasks)
