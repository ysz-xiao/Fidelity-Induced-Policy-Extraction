import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib as plt

def plot_confusion_matrix(y_pred, y_real, FILE_PATH):
    """
    画混淆矩阵
    输入：
    y_pred   预测y值，list[y1,y2....]
    y_real   真实y值，list[y1,y2....]
    FILE_PATH    图片存储路径
    """
    FILE_PATH = "{}".format(FILE_PATH)  # e.g., "XAI_results/confusing_matrix.png"
    labels = [range(9)]
    conf_mat = confusion_matrix(list(y_real), list(y_pred), labels=labels)
    df = pd.DataFrame(conf_mat)
    fig = sns.heatmap(df, annot=False, cmap="YlGnBu")
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(FILE_PATH, dpi=400)


def text_representation(clf):
    """
    打印数据决策树的文本形式
    输入：
    clf   决策树，只适用于硬决策树
    """
    from sklearn import tree
    text_representation = tree.export_text(clf)
    print(text_representation)


def plot_feature_importance(clf, feature_names):
    """
    画特征重要性图，按重要性达到小排列
    输入：
    clf  模型
    feature_names 特征名列表 list["name1","name2"...]
    """
    feature_importance = np.abs(np.sum(clf.coef_, axis=0))
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    # plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    # plt.yticks(pos, sorted_idx)
    plt.yticks(pos, [feature_names[idx] for idx in sorted_idx])
    plt.xlabel('Relative Importance')
    plt.show()
