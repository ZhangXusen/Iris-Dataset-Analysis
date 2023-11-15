# 数据集
from sklearn import datasets
# 分类器
from sklearn import tree
# 训练集测试集分割模块
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 自定义导入数据集函数
def get_data(total_data):
    # 显示total_data包含的内容
    print("传入数据集包含内容有：", [x for x in total_data.keys()])
    # 样本
    x_true = total_data.data
    # 标签
    y_true = total_data.target
    # 特征名称
    feature_names = total_data.feature_names
    # 类名
    target_names = total_data.target_names

    return x_true, y_true, feature_names, target_names


# 定义主函数
def main():
    # 利用自定义函数导入Iris数据集
    total_iris = datasets.load_iris()
    x_true, y_true, feature_names, target_names = get_data(total_iris)
    # 分割数据集
    rate_test = 0.2  # 训练集比例
    x_train, x_test, y_train, y_test = train_test_split(x_true,
                                                        y_true,
                                                        test_size=rate_test)
    print("\n训练集样本大小：", x_train.shape)
    print("训练集标签大小：", y_train.shape)
    print("测试集样本大小：", x_test.shape)
    print("测试集标签大小：", y_test.shape)

    # 设置决策树分类器
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    # 训练模型
    clf.fit(x_train, y_train)
    # 评价模型
    score = clf.score(x_test, y_test)
    print("\n模型测试集准确率为：", score)
    # 显示特征重要程度
    print("\n特征重要程度为：")
    info = [*zip(feature_names, clf.feature_importances_)]
    for cell in info:
        print(cell)


# 调用主函数
if __name__ == "__main__":
    main()
