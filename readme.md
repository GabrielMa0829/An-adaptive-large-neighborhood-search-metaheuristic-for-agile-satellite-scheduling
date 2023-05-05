# 使用方法
直接运行main函数即可

函数、类等定义都在task文件中

# 可以手动修改的部分
全局定义的所有变量均为可手动修改的参数

# 目前的配置
1. 所有的任务都是随机函数生成，每次运行的任务数据全部一致
2. 所有任务的VTW开始时间随机生成，但VTW的长度一致
3. 目前只有一段观测轨道，若涉及到多段轨道，里面大多数的函数需要修改（因为任务数据大多数定义的都是二维列表，若涉及到多段轨道，则需要改为三维列表）
4. 确定任务开始的时间，在可用时间的开始时刻
5. 任务所有的可占内存均为4，还未编写存储量的约束条件

# 每个文件的作用
aaa为测试文件
function 为测试函数 常用来编写一些测试内容
main 为主函数 50组任务 每组执行10次 并生成结果保存
One 某一特定任务组执行一次的结果 附带画图
profit_data 保存main运行结果的利润
task.py 任务文件