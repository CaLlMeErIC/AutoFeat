import os
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LightGBMModel:
    def __init__(self, max_depth=10, objective='binary', n_estimators=100000, learning_rate=0.01):
        self.model = LGBMClassifier(
            max_depth=max_depth,
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        self.is_trained = False

    @staticmethod
    def load_data_from_files(x_path, y_path):
        """
        从指定的特征和标签文件加载数据。
        """
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError("特征文件或标签文件未找到。")
        X = np.load(x_path)
        y = np.load(y_path)
        return X, y

    @staticmethod
    def load_data_from_folder(folder_path):
        """
        从指定文件夹加载所有 .npy 文件并拼接成特征和标签。
        文件名应为类似于 '1.npy', '2.npy' 等。
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError("指定的路径不是文件夹。")

        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 按文件名中的数字排序

        data_list = []
        for file in files:
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            data_list.append(data)

        data_array = np.concatenate(data_list, axis=0)
        return data_array

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        训练模型。
        """
        eval_set = [(x_train, y_train)]
        if x_valid is not None and y_valid is not None:
            eval_set.append((x_valid, y_valid))

        self.model.fit(
            x_train, y_train,
            eval_set=eval_set,
            # binary_logloss
            # multi_logloss,multi_error,cross_entropy
            eval_metric="binary_logloss",
            callbacks=[
                lgb.log_evaluation(period=1),
                lgb.early_stopping(stopping_rounds=300)
            ]
        )
        self.is_trained = True

    def save_model(self, model_path):
        """
        保存模型到指定路径。
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存。")
        joblib.dump(self.model, model_path)
        print(f"模型已保存到 {model_path}")

    def load_model(self, model_path):
        """
        从指定路径加载模型。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("模型文件未找到。")
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"模型已从 {model_path} 加载。")

    def predict(self, X):
        """
        使用模型进行预测。
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法预测。")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        使用模型进行概率预测。
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法预测。")
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type='split', max_num_features=None, feature_name_mapping=None):
        """
        输出特征重要性。

        :param importance_type: 特征重要性类型，可选 'split'（默认）或 'gain'。
        :param max_num_features: 可选，输出的重要性特征的最大数量。
        :param feature_name_mapping: 可选，特征名到索引的映射字典，格式为 {特征名: 索引}。
        :return: 特征重要性的列表。
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法获取特征重要性。")
        if importance_type == 'split':
            importance = self.model.feature_importances_
        elif importance_type == 'gain':
            importance = self.model.booster_.feature_importance(importance_type='gain')
        else:
            raise ValueError("importance_type 只能是 'split' 或 'gain'")

        feature_importance = {}
        for idx, score in enumerate(importance):
            if feature_name_mapping:
                # 构建索引到特征名的映射
                inverted_mapping = {v: k for k, v in feature_name_mapping.items()}
                feature_name = inverted_mapping.get(idx, f'feature_{idx}')
            else:
                feature_name = f'feature_{idx}'
            feature_importance[feature_name] = score

        # 按重要性排序
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        if max_num_features:
            sorted_features = sorted_features[:max_num_features]

        # 输出特征重要性
        print("Feature Importances:")
        for feature, score in sorted_features:
            print(f"Feature: {feature}, Importance: {score}")

        return sorted_features


if __name__ == "__main__":
    # 初始化模型
    model = LightGBMModel()

    # 从指定的特征和标签文件加载数据（只有一个数据集）
    X, y = model.load_data_from_files('data.npy', 'labels.npy')

    # 将数据集划分为训练集和测试集
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 训练模型
    model.train(x_train, y_train, x_valid, y_valid)

    # 保存模型
    model.save_model('lightgbm_model.joblib')

    # 如果需要，可以加载模型
    # model.load_model('lightgbm_model.joblib')

    # 进行预测
    predictions = model.predict(x_valid)
    probabilities = model.predict_proba(x_valid)

    # 特征名到索引的映射
    feature_name_mapping = {}
    # 打开并读取 JSON 文件
    with open('feature_indices.json', 'r', encoding='utf-8') as file:
        feature_name_mapping = json.load(file)

    # 输出特征重要性，传入特征名映射字典
    model.get_feature_importance(feature_name_mapping=feature_name_mapping)
    # 修改特征重要性调用并添加绘图代码
    # 获取前10个重要特征
    sorted_features = model.get_feature_importance(
        feature_name_mapping=feature_name_mapping,
        max_num_features=10,
        importance_type='gain'  # 可以切换为'split'或'gain'
    )

    # 绘制特征重要性柱状图
    plt.figure(figsize=(10, 6))
    features = [f[0] for f in sorted_features]
    importance_scores = [f[1] for f in sorted_features]

    # 创建水平柱状图
    plt.barh(range(len(features)), importance_scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()  # 将最重要的特征显示在顶部
    plt.tight_layout()

    # 保存图片
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("特征重要性图已保存为 feature_importance.png")
    plt.show()
    plt.close()
