def create_dir(DIR):
    import os
    if not os.path.exists(DIR):  # 创建路径
        os.makedirs(DIR)

def write_csv(title, dataset, path, filename, auto_cal=False,is_float=False):
    """写入csv表格数据
    Parameters
    ----------
    title 数据标题
    dataset 数据内容
    path 后缀 .csv
    auto_cal 是否自动计算均值、最大值、最小值

    Returns
    -------

    """
    import csv
    import numpy
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(path+"/"+filename+".csv", 'w', encoding="utf-8", newline="")
    writer = csv.writer(file)
    if auto_cal:
        # 计算平均值
        writer.writerow(["avg:"] + [""] * (len(title) - 1))
        writer.writerow(numpy.mean(dataset, axis=0))
        # 计算计算最大值
        writer.writerow(["max:"] + [""] * (len(title) - 1))
        writer.writerow(numpy.max(dataset, axis=0))
        # 计算计算最小值
        writer.writerow(["min:"] + [""] * (len(title) - 1))
        writer.writerow(numpy.min(dataset, axis=0))
    writer.writerow(title)
    for data in dataset:
        writer.writerow(data)

    file.close()
