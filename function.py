from task import *
from main import *

if __name__ == "__main__":
    Current_solution = Platform(num_schedule, num_task, num_task_info, 100)
    Current_solution.produce_schedule()
    Current_solution.produce_tasks(num_task, num_task_info)
    print(Current_solution.tasks)

