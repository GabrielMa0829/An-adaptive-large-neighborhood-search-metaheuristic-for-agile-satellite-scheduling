# import xlrd

from pylab import mpl
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from numpy import exp
from scipy import stats

# random.seed(100)
# np.random.seed(100)

"""
处理原始数据，产生一个卫星目标点数组和一个任务列表数组
"""


# n_targets 调度序列的个数
# n_task 任务个数  任务列表的行数
# n_stacked_observation 任务的属性个数 人物列表的列数
class Platform(object):
    def __init__(self, n_targets, n_task, n_stacked_observation, random_seed):
        self.tasks = []  # 任务列表
        self.target = []  # 空余时间
        self.solution = []  # 用于存放解  调度顺序 里面存放的是任务的编号
        self.n_targets = n_targets
        self.n_task = n_task
        self.n_stacked_observation = n_stacked_observation
        self.trans = 2  # 最小转换时间
        self.sample = stats.poisson.rvs(mu=10, size=n_task, random_state=10)  # 每10到达一个任务  存储为(1,50)的列表
        # self.list_q = []  # 用于存放被destroy拆解下来的任务
        self.list_f = [i + 1 for i in range(n_task)]  # 用于存放未安排的任务
        # self.solution_info = []  # 存放解中任务的详细信息
        self.seed = random_seed
        random.seed(self.seed)  # 设置种子值使每次生成的随机数都相同  每次调用随机数都要设置相同的种子才能生成成相同的随机数
        # 因为写在了构造函数当中，所以类中每次赋值都会自动设置seed
        # np.random.seed(seed)  #这句注释后无变化 所以就注释了
        # self.sample = stats.poisson.rvs(mu=50, size=100, random_state=10)
        # self.profit = stats.poisson.rvs(mu=5, size=50, random_state=10)
        # self.profit = np.random.uniform(low=1, high=10, size=50).tolist()

    # 处理原始数据    调度序列
    def produce_schedule(self):
        self.target = [[0 for _ in range(4)] for _ in range(80)]
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
            self.tasks[i][1] = random.randint(0, 600)  # VTW开始时间
            last_time = 599  # 任务持续时间  VTW长度
            self.tasks[i][2] = self.tasks[i][1] + last_time  # 任务结束观测时间
            self.tasks[i][3] = random.randint(10, 90)  # 观测时间 10~90s
            self.tasks[i][4] = random.randint(1, 10)  # 观测收益  1~10
            self.tasks[i][5] = 4  # 所占内存
            if i == 0:
                self.tasks[i][6] = self.sample[i]  # 任务到达时间
            else:
                self.tasks[i][6] = self.sample[i] + self.tasks[i - 1][6]
            # 设置random_state时，每次生成的随机数一样。不设置或为None时，多次生成的随机数不一样
        max_value = self.tasks[n_task-1][6]
        for i in range(n_task):
            if max_value > self.tasks[i][1]:
                self.tasks[i][1] = max_value
        return self.tasks


# 计算对应调度的利润profit
def profits(platform):
    profit = 0
    for i in platform.solution:
        if i != 0:
            profit += platform.tasks[i - 1][4]
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
    if platform.target[loc][2] - platform.target[loc][1] != platform.target[loc][3] or platform.target[loc + 1][2] - \
            platform.target[loc + 1][1] != platform.target[loc + 1][3]:
        print("error")
    platform.solution.insert(loc, task_id)
    platform.list_f.remove(task_id)  # 在候选任务中删除已经安排过的任务 避免重复接受任务
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
    loc = platform.solution.index(task_id)  # 读取需要删除的任务在解中的索引
    platform.target[loc][2] = platform.target[loc + 1][2]  # 对时间窗口进行更改
    platform.target[loc][3] = platform.target[loc][2] - platform.target[loc][1]
    platform.target.pop(loc + 1)
    platform.solution.remove(task_id)
    platform.list_f.append(task_id)  # 将删除的任务索引存到 Q列表


# 判断插入的位置  输入对象、需要插入的任务id  即可输出可以插入的位置
def insert_location(platform, task_id):
    """

    :param platform: 对象
    :param task_id: 任务编号
    :return: 如果能插入返回插入的索引，否则返回-1
    """
    vtw_start = platform.tasks[task_id - 1][1]  # 任务的VTW的开始时间
    vtw_end = platform.tasks[task_id - 1][2]  # 任务的VTW的结束时间
    available_time = 0  # 可用的空闲时间
    loc = 0
    for i in platform.target:  # 寻找可用时间
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
            if available_time >= platform.tasks[task_id - 1][3] + 2 * platform.trans:
                break
    if available_time >= platform.tasks[task_id - 1][3] + 2 * platform.trans:
        return loc + platform.trans
    else:
        return -1


# 返回解中任务的详细信息列表
def task_info(p, l1):
    info = []
    for i in l1:
        info.append(p.tasks[i - 1])
    return info


#  随机移除 编号0
def destroy_random(platform):
    random.seed()  # 重置种子 使得每次随机的索引都不一样
    remove_index = random.sample(platform.solution, q)
    for i in remove_index:
        delete_task(platform, i)
    return 0


# 最小收益优先移除：先将解中任务排序后依次移除q个任务 编号1
def destroy_min_profit(p: Platform):
    task = task_info(p, p.solution)
    count = 0
    task = sorted(task, key=(lambda x: x[4]))  # 将解的任务列表按照利润升序排序
    # print(task)
    for i in task:
        delete_task(p, i[0])
        count += 1
        if count == 6:
            break


# 冲突移除：计算解中每个任务与候选任务的冲突度，按照冲突度降序依次delete  编号2
def destroy_max_conflict(p: Platform):
    task1 = task_info(p, p.solution)
    task2 = task_info(p, p.list_f)
    conflict = conflict_degree(task1, task2)
    conflict = sorted(conflict, key=lambda x: x[1], reverse=True)
    # print(conflict)
    for i in range(q):
        delete_task(p, conflict[i][0])
    # print(p.solution)


def performance(p: Platform, li):
    perf = []
    for i in li:
        perf2 = p.tasks[i - 1][4] / p.tasks[i - 1][3]
        perf.append([i, perf2])
    return perf


# 性价比删除：优先删除最低性价比的任务  编号 3
def destroy_performance(p: Platform):
    perf = performance(p, p.solution)
    perf = sorted(perf, key=(lambda x: x[1]))  # 性价比升序排序
    # print(perf)
    for i in range(q):
        delete_task(p, perf[i][0])
    return 0


# 贪婪插入 编号0
def repair_greedy(p: Platform):
    task = task_info(p, p.list_f)
    task = sorted(task, key=lambda x: x[4], reverse=True)
    for i in task:
        insert_loc = insert_location(p, i[0])
        if insert_loc > 0:
            insert_task(p, i[0], insert_loc)


#  最小冲突插入 编号1
def repair_min_conflict(p: Platform):
    task_s = task_info(p, p.solution)  # 获得解中任务的信息列表 以及候选任务的信息列表
    task_f = task_info(p, p.list_f)
    conflict = conflict_degree(task_f, task_s)  # 计算候选任务中的每个任务与解中任务的冲突度
    conflict = sorted(conflict, key=lambda x: x[1])  # 冲突度升序排序
    for i in conflict:  # 依次尝试插入 遍历完毕后即为插入成功
        insert_loc = insert_location(p, i[0])
        if insert_loc > 0:
            insert_task(p, i[0], insert_loc)


# 性价比插入：候选任务按照性价比降序排列 依次尝试插入即可 编号2
def repair_performa(p: Platform):
    perf = performance(p, p.list_f)
    perf = sorted(perf, key=lambda x: x[1], reverse=True)
    for i in perf:
        insert_loc = insert_location(p, i[0])
        if insert_loc > 0:
            insert_task(p, i[0], insert_loc)


# 解的初始化  使用贪婪启发式算法
# 按照收益的降序和开始时间的升序进行排列，依次地尝试将每个VTW插入到当前地调度中，所有的VTW访问结束则初始化完毕
def init_solution(platform):
    task = sorted(platform.tasks, key=(lambda x: (x[4], -x[1])), reverse=True)
    for i in task:
        loc = insert_location(platform, i[0])
        if loc > 0:
            insert_task(platform, i[0], loc)
            # platform.list_f.remove(i[0])


# 计算冲突度，输入两个任务列表，计算l1中每个元素与l2中元素的冲突度
def conflict_degree(l1, l2):
    c_d = []  # 记录l1中所有任务的冲突度情况
    for i in l1:  # 依次计算l1中每个任务与l2中所有任务的冲突度
        num = 0  # 冲突个数
        length = 0  # 冲突长度总和
        for j in l2:  # 用l1中一个任务与l2中所有任务做冲突度计算
            con = 0
            if i[1] < j[2] and i[2] > j[1]:  # 判断时间窗口是否有重叠？没有重叠则下一次循环
                if i[1] < j[1]:
                    if i[2] < j[2]:
                        con = i[2] - j[1]
                    else:
                        con = j[3]
                else:
                    if i[2] < j[2]:
                        con = i[2] - i[1]
                    else:
                        con = j[2] - i[1]
            if con != 0:
                num += 1
                length += con
        if num != 0:
            con2 = length / num
        else:
            con2 = 0
        c_d.append([i[0], con2])
    return c_d


# 轮盘赌 选择destroy方法
def select_destroy(p: Platform):
    destroyRoulette = np.array(wDestroy).cumsum()  # 轮盘赌 cumsum()把列表里之前数的和加到当前列 eg. [1,2,3,4] cumsum结果为[1,3,6,10]
    random.seed()
    r = random.uniform(0, max(destroyRoulette))  # 随机生成 0 - 轮盘赌列表中的最大值  之间的浮点数
    # print("随机数{}".format(r))
    for i in range(len(destroyRoulette)):
        if r <= destroyRoulette[i]:
            if i == 0:
                destroy_random(p)
            elif i == 1:
                destroy_min_profit(p)
            elif i == 2:
                destroy_max_conflict(p)
            else:
                destroy_performance(p)
            return i


# 轮盘赌 选择repair方法
def select_repair(p: Platform):
    repairRoulette = np.array(wRepair).cumsum()
    random.seed()
    r = random.uniform(0, max(repairRoulette))
    for i in range(len(repairRoulette)):
        if r <= repairRoulette[i]:
            if i == 0:
                repair_greedy(p)
            elif i == 1:
                repair_min_conflict(p)
            else:
                repair_performa(p)
            return i


def update(p1: Platform, p2: Platform, p3: Platform, destroy_index, repair_index):
    """
    根据输入的当前解和改造后的新解决定是否用新解替换当前解，并更新分数
    :param p3: 最优解
    :param p1: 当前解
    :param p2: 新解
    :param destroy_index:使用的destroy方法的索引
    :param repair_index: 使用的repair方法的索引
    """
    destroy_use_times[destroy_index] += 1  # 对应次数+1
    repair_use_times[repair_index] += 1
    profit1 = profits(p1)  # 当前解的利润
    profit2 = profits(p2)  # 新解的利润
    profit3 = profits(p3)  # 最优解的利润
    if profit2 >= profit1:  # 如果新解的利润大于等于当前解的利润 则 新解替换当前解
        p1 = copy.deepcopy(p2)
        if profit2 >= profit3:
            # print("New:{}  Best:{}".format(profit2, profit3))
            p3 = copy.deepcopy(p2)
            destroy_score[destroy_index] += round(update_standard[0], 5)
            repair_score[repair_index] += round(update_standard[0], 5)
        else:
            destroy_score[destroy_index] += round(update_standard[1], 5)
            repair_score[repair_index] += round(update_standard[1], 5)
    else:  # 新解利润小于当前解 利用退火算法接受准则计算 判断 是否接受该解
        r = random.random()
        if r < exp((100 / T) * (profit2 - profit1) / profit1):  # 温度越低 接受差解的概率越低
            p1 = copy.deepcopy(p2)
            destroy_score[destroy_index] += round(update_standard[2], 5)
            repair_score[repair_index] += round(update_standard[2], 5)
        else:
            destroy_score[destroy_index] += round(update_standard[3], 5)
            repair_score[repair_index] += round(update_standard[3], 5)
    wDestroy[destroy_index] = round(
        (1 - b) * wDestroy[destroy_index] + b * destroy_score[destroy_index] / destroy_use_times[
            destroy_index], 5)
    wRepair[repair_index] = round((1 - b) * wRepair[repair_index] + b * repair_score[repair_index] / repair_use_times[
        repair_index], 5)
    Best_prof.append(profit3)
    C_prof.append(profit1)
    return p1, p2, p3  # 因为这里的赋值使用的是deepcopy 可能导致了形参和实参不能对应的问题 所以要return一下参能返回给实参


def set_value(value, li):
    for i in range(len(li)):
        li[i] = value


def reset_weight():
    set_value(1, wDestroy)
    set_value(1, wRepair)
    set_value(0, destroy_use_times)
    set_value(0, repair_use_times)
    set_value(0, destroy_score)
    set_value(0, repair_score)


def picture_profit():
    mpl.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
    plt.figure(1, figsize=(10, 8), dpi=150)
    plt.title('profits')
    plt.xlabel('迭代次数')  # 为x轴命名为“x”
    plt.ylabel('利润')  # 为y轴命名为“y”
    plt.xlim(0, len(Best_prof))  # 设置x轴的范围为
    plt.ylim(200, 400)  # 同上
    plt.plot(range(1, len(Best_prof) + 1), Best_prof, c='red')
    plt.plot(range(1, len(Best_prof) + 1), C_prof, c='blue')
    x_major_locator = MultipleLocator(1000)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.legend(['最高利润', '当前利益'])  # 图例
    plt.show()  # 显示

    plt.figure(2, figsize=(10, 8), dpi=150)
    plt.title('iterx_prof')
    plt.xlabel('迭代次数')  # 为x轴命名为“x”
    plt.ylabel('每次迭代结束后的利润')  # 为y轴命名为“y”
    plt.xlim(0, len(iterx_prof))  # 设置x轴的范围为
    plt.ylim(200, 400)  # 同上
    plt.plot(range(1, len(iterx_prof) + 1), Best_iterx_prof, c='red')
    plt.plot(range(1, len(iterx_prof) + 1), iterx_prof, c='blue')
    x_major_locator = MultipleLocator(10)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.legend(['最高利润', '当前利益'])  # 图例
    plt.show()  # 显示


num_schedule = 1  # 调度序列的个数
num_task = 50  # 任务个数
num_task_info = 7  # 任务属性个数
iterx, iterxMax = 0, 500  # 初始迭代次数、最大迭代次数
Best_prof = []  # 记录最高利润（画图用）
C_prof = []  # 当前利润（画图用）
iterx_prof = []  # 记录每次迭代结束后的利润（画图用）
Best_iterx_prof = []
b = 0.5  # 更新权重的参数（控制权重变化速度）
q = 6  # 任务库容量
T = 100  # 初始温度
Pa = 0.97  # 降温指数
# num_schedule = 1  # 调度序列的个数
# num_task = 100  # 任务个数
# num_task_info = 6  # 任务属性个数
update_standard = [1.5, 1.2, 0.8, 0.6]
wDestroy = [1. for _ in range(4)]  # 摧毁算子的初始权重，[1,1]
wRepair = [1. for _ in range(3)]  # 修复算子的初始权重
destroy_use_times = [0 for _ in range(4)]  # 摧毁初始次数，0
repair_use_times = [0 for _ in range(3)]  # 修复初始次数
destroy_score = [0 for _ in range(4)]  # 摧毁算子初始得分
repair_score = [0 for _ in range(3)]  # 修复算子初始得分

# if __name__ == '__main__':
#     Current_solution = Platform(num_schedule, num_task, num_task_info)
#     Current_solution.produce_schedule()
#     Current_solution.produce_tasks(num_task, num_task_info)
#
#     # print(a.target)
#     # print(a.tasks)
#     # print(a.solution)
#     init_solution(Current_solution)
#     New_solution = Best_solution = copy.deepcopy(Current_solution)
#     print(Current_solution.target)
#     print(Current_solution.solution)
#     print("profits:{}".format(profits(Current_solution)))
#     destroy_random(New_solution)
#     print(New_solution.solution)
#     repair_min_conflict(New_solution)
#     print("profits:{}".format(profits(New_solution)))
