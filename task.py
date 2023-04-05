# import xlrd
import random


# random.seed(100)
# np.random.seed(100)

"""
处理原始数据，产生一个卫星目标点数组和一个任务列表数组
"""


# n_targets 调度序列的个数
# n_task 任务个数  任务列表的行数
# n_stacked_observation 任务的属性个数 人物列表的列数
class Platform(object):
    def __init__(self, n_targets, n_task, n_stacked_observation):
        self.tasks = []  # 任务列表
        self.target = []  # 空余时间
        self.solution = []  # 用于存放解  调度顺序 里面存放的是任务的编号
        self.n_targets = n_targets
        self.n_task = n_task
        self.n_stacked_observation = n_stacked_observation
        self.trans = 5  # 最小转换时间
        seed = 101
        random.seed(seed)  # 设置种子值使每次生成的随机数都相同  每次调用随机数都要设置相同的种子才能生成成相同的随机数
        # 因为写在了构造函数当中，所以类中每次赋值都会自动设置seed
        # np.random.seed(seed)  #这句注释后无变化 所以就注释了
        # self.sample = stats.poisson.rvs(mu=50, size=100, random_state=10)
        # self.profit = stats.poisson.rvs(mu=5, size=50, random_state=10)
        # self.profit = np.random.uniform(low=1, high=10, size=50).tolist()

    # 处理原始数据    调度序列
    def produce_schedule(self):
        self.target = [[0 for _ in range(4)] for _ in range(50)]
        # self.target = target_temp
        for i in range(1):
            t = i
            self.target[t][0] = 0  # 任务编号 若为0则此处空闲
            self.target[t][1] = 0  # 开始观测时间,将时间换为秒
            self.target[t][2] = 1200  # 结束观测时间
            self.target[t][3] = 0  # 持续观测时间
            # self.target[t][4] = '1'  # 不用管
            # self.target[t][5] = random.randint(1,10) #收益大小
        return self.target

    # 用于创建任务，任务数目为n_tasks，N_TASK为每行的维度
    def produce_tasks(self, n_task, n_stacked_observation):
        self.tasks = [[0 for _ in range(n_stacked_observation)] for _ in range(n_task)]
        for i in range(n_task):
            self.tasks[i][0] = i + 1  # 任务编号
            self.tasks[i][1] = random.randint(0, 600 * (self.n_task / 50))  # 任务开始观测时间
            last_time = 600  # 任务持续时间  VTW长度
            self.tasks[i][2] = self.tasks[i][1] + last_time  # 任务结束观测时间
            self.tasks[i][3] = random.randint(20, 80)  # 观测时间 50s
            self.tasks[i][4] = random.randint(1, 10)  # 观测收益
            self.tasks[i][5] = 4  # 所占内存
            # 设置random_state时，每次生成的随机数一样。不设置或为None时，多次生成的随机数不一样
            # self.tasks[i][6] = self.sample[i]  # 任务到达时间  不管
            # if self.tasks[i][6] > self.tasks[i][1]:
            #     self.tasks[i][6] = self.tasks[i][1] - 1
        return self.tasks


# 计算对应调度的利润profit
def profits(platform):
    profit = 0
    for i in platform.solution:
        if i != 0:
            profit += platform.tasks[i][4]
    return profit


# 插入函数：将待插入的节点插入到对应位置的 操作  需要输入platform，待插入的任务编号，插入位置。
def insert_task(platform, task_id, insert_loc):
    loc = 0
    for i in platform.target:
        if i[2] >= insert_loc:
            loc = platform.target.index(i)
            break
    # 插入一个元素 [30,60]->[0,100] 插入后 [[0,30],[60,100]]
    # [80,100]->[[20,60],[70,100]] 插入后 [[20,60],[70,80],[100,100]]
    platform.target.insert(loc, [platform.target[loc][0], platform.target[loc][1], insert_loc,
                                 insert_loc - platform.target[loc][1]], )
    platform.target[loc + 1][1] = insert_loc + platform.tasks[task_id - 1][3]
    platform.target[loc + 1][3] = platform.target[loc + 1][2] - platform.target[loc + 1][1]
    platform.solution.insert(loc, task_id)
    # # 插入一共分四种情况  (弃用的方法)
    # # 第一种 插入的任务在可用时间之内 不等于边界
    # if insert_loc > platform.target[loc][1] and \
    #         insert_loc + platform.tasks[task_id - 1][3] < platform.target[loc][2]:
    #     platform.target.insert(loc, [platform.target[loc][0], platform.target[loc][1], insert_loc,
    #                                  insert_loc - platform.target[loc][1]], )
    #     # 插入一个元素 [30,60]->[0,100] 插入后 [[0,30],[60,100]]
    #     platform.target[loc + 1][1] = insert_loc + platform.tasks[task_id - 1][3]
    #     platform.target[loc + 1][3] = platform.target[loc + 1][2] - platform.target[loc + 1][1]
    # # 第二种 插入的任务，任务开始时间 等于 可用时间的开始时间 [0,30]->[0,100]  插入后  [[30,100]]
    # elif insert_loc == platform.target[loc][1] and \
    #         insert_loc + platform.tasks[task_id - 1][3] < platform.target[loc][2]:
    #     platform.target[loc][1] = insert_loc + platform.tasks[task_id - 1][3]
    # # 第三种 插入的任务，任务结束时间 等于 可用时间的结束时间  [70,100]->[0,100]  插入后  [[0,70]]
    # elif insert_loc > platform.target[loc][1] and \
    #         insert_loc + platform.tasks[task_id - 1][3] == platform.target[loc][2]:
    #     platform.target[loc][2] = insert_loc
    # # # 第四种 插入的任务，任务开始和结束时间 恰好等于 可用时间的开始和结束时间  [30,60]->[[0,20],[30,60],[70,100]] 插入后 [[0,20],[70,100]]
    # # # 不存在这种情况，任务和任务之间应该存在最小转换时间Trans
    # # elif insert_loc == platform.target[loc][1] and \
    # #         insert_loc + platform.tasks[task_id - 1][3] == platform.target[loc][2]:
    # #     platform.target.pop(loc)
    # else:
    #     print("插入出现错误！")
    # platform.solution.insert(loc, task_id)


# 删除函数：输入待删除的任务编号
def delete_task(platform, task_id):
    loc = platform.solution.index(task_id)
    platform.target[loc][2] = platform.target[loc + 1][2]
    platform.target.pop(loc + 1)
    platform.solution.remove(task_id)


# 判断插入的位置  输入对象、需要插入的任务id  即可输出可以插入的位置
def insert_location(platform, task_id):
    vtw_start = platform.tasks[task_id - 1][1]
    vtw_end = platform.tasks[task_id - 1][2]
    available_time = 0
    loc = 0
    for i in platform.target:
        if vtw_start < i[2] and vtw_end > i[1]:
            if vtw_start < i[1]:
                loc = i[1]
                if vtw_end < i[2]:
                    available_time = vtw_end - i[1]
                else:
                    available_time = i[3]
            else:
                loc = vtw_start
                if vtw_end < i[2]:
                    available_time = vtw_end - vtw_start
                else:
                    available_time = i[2] - vtw_start
            break
    if available_time >= platform.tasks[task_id - 1][3] + platform.trans:
        return loc + platform.trans
    else:
        return -1


# 解的初始化  使用贪婪启发式算法
# 按照收益的降序和开始时间的升序进行排列，依次地尝试将每个VTW插入到当前地调度中，所有的VTW访问结束则初始化完毕
def init_solution(platform):
    task = sorted(platform.tasks, key=(lambda x: (x[4], -x[1])), reverse=True)
    print(task)
    for i in task:
        loc = insert_location(platform, i[0])
        if loc > 0:
            insert_task(platform, i[0], loc)


if __name__ == '__main__':
    num_schedule = 1  # 调度序列的个数
    num_task = 100  # 任务个数
    num_task_info = 6  # 任务属性个数

    a = Platform(num_schedule, num_task, num_task_info)
    a.produce_schedule()
    a.produce_tasks(num_task, num_task_info)
    print("")
    print(a.target)
    print(a.tasks)
    print(a.solution)
    init_solution(a)
    print(a.target)
    print(a.solution)
    print("profits:{}".format(profits(a)))
