import pandas as pd
import numpy as np
import json
import os
import shutil
from datetime import datetime


class FeatureExtractor:
    def __init__(self, file_path, exclude_cols=None, label_col=None, id_col=None, drop_first_col=False,
                 split_date=False,
                 existing_mapping_path=None
                 ):
        """
        初始化方法。

        :param file_path: CSV或Excel文件的路径。
        :param exclude_cols: 需要排除的列名列表。
        :param label_col: 标签列的列名。
        :param id_col: 用作ID的列名。
        :param drop_first_col: 是否删除第一列，默认不删除。
        :param split_date: 是否拆分日期。
        :param existing_mapping_path: 是否存在生产过的特征字典。
        """
        self.file_path = file_path
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.label_col = label_col
        self.id_col = id_col
        self.drop_first_col = drop_first_col
        self.df = None
        self.features = None
        self.labels = None
        self.ids = None  # 存储ID列
        self.feature_names = None
        self.mapping_dicts = {}  # 存储分类变量的映射字典
        self.split_date = split_date  # 是否拆分日期
        self.feature_name_to_index = {}  # 存储特征名与其对应的列索引
        self.existing_mapping_dicts = self._load_existing_mappings(existing_mapping_path)  # 加载现有映射

    @staticmethod
    def _load_existing_mappings(path):
        """加载现有的特征映射字典"""
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def read_file(self):
        """
        读取CSV或Excel文件，并根据需要删除第一列。
        """
        # 获取文件扩展名
        file_ext = os.path.splitext(self.file_path)[1].lower()
        # 根据扩展名选择读取方法
        if file_ext == '.csv':
            self.df = pd.read_csv(self.file_path)
        elif file_ext in ['.xlsx', '.xls']:
            self.df = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"不支持的文件格式：{file_ext}")

        print(f"已读取文件：{self.file_path}")

        # 删除第一列（如果需要）
        if self.drop_first_col and self.df.shape[1] > 1:
            self.df = self.df.iloc[:, 1:]
            print("已删除第一列")

    def extract_features_and_labels(self):
        """
        提取特征、标签和ID，并处理不存在的列。
        """
        # 检查需要排除的列是否存在
        exclude_cols_existing = [col for col in self.exclude_cols if col in self.df.columns]
        # 如果提供了标签列，并且存在，则添加到排除列表中
        if self.label_col and self.label_col in self.df.columns:
            exclude_cols_existing.append(self.label_col)
        elif self.label_col:
            # 标签列不存在，输出提示信息
            print(f"警告：标签列 '{self.label_col}' 不存在，将不生成标签文件")
            self.label_col = None  # 重置标签列

        # 提取ID列（如果指定了id_col）
        if self.id_col and self.id_col in self.df.columns:
            self.ids = self.df[self.id_col]
            # 将id_col添加到排除列列表，防止其作为特征
            exclude_cols_existing.append(self.id_col)
            print(f"已指定ID列 '{self.id_col}'")
        else:
            # 如果未指定id_col，创建id序列
            self.ids = pd.Series(range(1, len(self.df) + 1), name='id')
            print("未指定ID列，使用从1开始的自增序列作为ID")

        # 提取特征：删除需要排除的列
        self.features = self.df.drop(columns=exclude_cols_existing, errors='ignore')

        # 提取标签（如果标签列存在）
        if self.label_col:
            self.labels = self.df[self.label_col]
        else:
            self.labels = None

        # 保存特征列名列表
        self.feature_names = self.features.columns.tolist()
        print(f"提取了特征列：{self.feature_names}")

    def process_mixed_types(self):
        """
        处理具有混合类型的列，将其统一转换为浮点型。
        """
        # 尝试将所有列转换为数值类型
        for col in self.features.columns:
            try:
                self.features[col] = pd.to_numeric(self.features[col], errors='raise')
            except:
                # 无法直接转换的列，跳过，留待后续处理
                continue

    def process_categorical_and_date_columns(self):
        """
        处理非数值类型的列：
        - 对于满足条件的分类变量，映射为整数。
        - 对于日期类型的列，拆分为对应的日期特征。
        """
        # 获取非数值类型的列
        non_numeric_cols = self.features.select_dtypes(exclude=['number']).columns.tolist()
        columns_to_drop = []
        for col in non_numeric_cols:
            col_data = self.features[col]
            # 尝试解析为日期类型
            parsed_dates = pd.to_datetime(col_data, errors='coerce', infer_datetime_format=True)
            if parsed_dates.notnull().sum() > 0 and self.split_date:
                # 是日期列，根据存在的日期部分拆分
                date_parts = {'year': parsed_dates.dt.year,
                              'month': parsed_dates.dt.month,
                              'day': parsed_dates.dt.day,
                              'hour': parsed_dates.dt.hour}

                # 检查各日期部分的值是否全为NaN
                existing_parts = []
                for part in date_parts:
                    if date_parts[part].notnull().sum() > 0:
                        existing_parts.append(part)

                # 根据存在的日期部分创建特征
                for part in existing_parts:
                    part_series = date_parts[part].fillna(-1).astype(int)
                    # 检查该日期部分的值是否全为0或-1
                    if part_series.isin([0, -1]).all():
                        print(f"列 '{col}' 的 '{part}' 部分全为0或缺失，跳过该特征")
                        continue  # 跳过该日期部分
                    self.features[f"{col}_{part}"] = part_series
                    print(f"已为列 '{col}' 添加特征 '{col}_{part}'")
                    self.feature_names.append(f"{col}_{part}")

                # 删除原始日期列
                self.features.drop(columns=[col], inplace=True)
                if col in self.feature_names:
                    self.feature_names.remove(col)
            else:
                # 非日期列，处理分类变量
                if col in self.existing_mapping_dicts:  # 存在现有映射
                    self._apply_existing_mapping(col)
                else:  # 不存在则创建新映射
                    self._create_new_mapping(col, columns_to_drop)
        # 删除无法处理的列
        if columns_to_drop:
            print(f"以下列无法处理，将被删除：{columns_to_drop}")
            self.features.drop(columns=columns_to_drop, inplace=True)
            self.feature_names = [fn for fn in self.feature_names if fn not in columns_to_drop]

    def _apply_existing_mapping(self, col):
        """应用现有的映射字典"""
        existing_mapping = self.existing_mapping_dicts[col]
        print(f"列 '{col}' 使用现有映射字典：{existing_mapping}")

        # 映射并处理未知值
        self.features[col] = (
            self.features[col]
            .map(lambda x: existing_mapping.get(str(x), -1))  # 处理类型差异
            .fillna(-1)
            .astype(int)
        )
        # 保留现有映射
        self.mapping_dicts[col] = existing_mapping

    def _create_new_mapping(self, col, columns_to_drop):
        """创建新的映射字典"""
        col_data = self.features[col]
        value_counts = col_data.value_counts(dropna=False)
        num_unique_values = len(value_counts)
        max_count = value_counts.max()

        if num_unique_values < 50 and max_count > 1:
            unique_values = [v for v in value_counts.index if pd.notnull(v)]
            mapping = {str(val): idx + 1 for idx, val in enumerate(unique_values)}  # 统一转为字符串键
            print(f"列 '{col}' 创建新映射字典：{mapping}")

            self.features[col] = (
                col_data.astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )
            self.mapping_dicts[col] = mapping
        else:
            columns_to_drop.append(col)

    def replace_nan(self):
        """
        将特征和标签数据中的NaN值替换为-1。
        """
        self.features.fillna(-1, inplace=True)
        if self.labels is not None:
            self.labels.fillna(-1, inplace=True)

    def generate_feature_name_to_index(self):
        """
        生成特征名与其对应的列索引的字典。
        """
        self.feature_name_to_index = {name: idx for idx, name in enumerate(self.features.columns)}

    def save_features_and_labels(self, output_numpy_array=False, output_row_features=False, removed_features_info=None):
        """
        将特征数据、标签数据和特征名分别保存到文件中。

        :param output_numpy_array: 是否将特征和标签保存为NumPy数组文件。
        :param output_row_features: 是否将每行特征和标签保存为单独的NumPy文件。
        """
        # 获取当前时间，格式为YYYYMMDDHHMM
        current_time = datetime.now().strftime('%Y%m%d%H%M')

        # 判断是否存在标签，决定文件夹命名
        if self.labels is not None:
            folder_name = f"features_labels_{current_time}"
        else:
            folder_name = f"features_{current_time}"

        # 创建输出文件夹
        output_folder = os.path.join('extract_result', folder_name)
        os.makedirs(output_folder, exist_ok=True)
        print(f"所有输出文件将保存到 '{output_folder}' 文件夹下")

        # 保存特征数据
        features_file = os.path.join(output_folder, 'features.csv')
        self.features.to_csv(features_file, index=False)
        print(f"特征数据已保存到 '{features_file}'")

        # 如果标签数据存在，保存标签数据
        if self.labels is not None:
            labels_file = os.path.join(output_folder, 'labels.csv')
            self.labels.to_csv(labels_file, index=False)
            print(f"标签数据已保存到 '{labels_file}'")

        # 保存ID列
        ids_file = os.path.join(output_folder, 'ids.csv')
        self.ids.to_csv(ids_file, index=False)
        print(f"ID数据已保存到 '{ids_file}'")

        # 保存特征名到文本文件
        feature_names_file = os.path.join(output_folder, 'feature_names.txt')
        with open(feature_names_file, 'w', encoding='utf-8') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        print(f"特征名已保存到 '{feature_names_file}'")

        # 保存特征名与列索引的字典
        feature_indices_file = os.path.join(output_folder, 'feature_indices.json')
        with open(feature_indices_file, 'w', encoding='utf-8') as f:
            json.dump(self.feature_name_to_index, f, ensure_ascii=False, indent=4)
        print(f"特征名与列索引的字典已保存到 '{feature_indices_file}'")

        # 保存分类变量的映射字典
        if self.mapping_dicts:
            mapping_dicts_file = os.path.join(output_folder, 'mapping_dicts.json')
            with open(mapping_dicts_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping_dicts, f, ensure_ascii=False, indent=4)
            print(f"分类变量的映射字典已保存到 '{mapping_dicts_file}'")

        # 额外的特征和标签输出方式

        # 方式一：保存所有特征和标签为NumPy数组
        if output_numpy_array:
            feature_array = self.features.to_numpy()
            features_npy_file = os.path.join(output_folder, 'data.npy')
            np.save(features_npy_file, feature_array)
            print(f"特征数据已保存为 NumPy 数组文件 '{features_npy_file}'")

            if self.labels is not None:
                labels_array = self.labels.to_numpy()
                labels_npy_file = os.path.join(output_folder, 'labels.npy')
                np.save(labels_npy_file, labels_array)
                print(f"标签数据已保存为 NumPy 数组文件 '{labels_npy_file}'")

        # 方式二：将每一行特征和标签以ID命名保存为单独的NumPy文件
        if output_row_features:
            features_folder = os.path.join(output_folder, 'features')
            os.makedirs(features_folder, exist_ok=True)
            feature_array = self.features.to_numpy()
            ids = self.ids.to_numpy()
            for idx in range(len(feature_array)):
                id_value = ids[idx]
                row_features = feature_array[idx]
                np.save(os.path.join(features_folder, f"{id_value}.npy"), row_features)
            print(f"每行特征已保存到 '{features_folder}' 文件夹下，以ID命名的NumPy文件")

            if self.labels is not None:
                labels_folder = os.path.join(output_folder, 'labels')
                os.makedirs(labels_folder, exist_ok=True)
                labels_array = self.labels.to_numpy()
                for idx in range(len(labels_array)):
                    id_value = ids[idx]
                    label_value = labels_array[idx]
                    # 保存单个标签为NumPy文件
                    np.save(os.path.join(labels_folder, f"{id_value}.npy"), label_value)
                print(f"每个标签已保存到 '{labels_folder}' 文件夹下，以ID命名的NumPy文件")

        # 如果标签数据存在，保存ID与标签的对应关系
        if self.labels is not None:
            labels_array = self.labels.to_numpy()
            ids = self.ids.to_numpy()
            labels_dict = dict(zip(ids.astype(str), labels_array.tolist()))
            id_label_file = os.path.join(output_folder, 'id_label.json')
            with open(id_label_file, 'w', encoding='utf-8') as f:
                json.dump(labels_dict, f, ensure_ascii=False, indent=4)
            print(f"ID与标签的对应关系已保存到 '{id_label_file}'")

        if removed_features_info:
            removed_features_file = os.path.join(output_folder, 'removed_features.json')
            with open(removed_features_file, 'w', encoding='utf-8') as f:
                json.dump(removed_features_info, f, ensure_ascii=False, indent=4)
            print(f"被移除特征信息已保存到 '{removed_features_file}'")

        # 复制训练py代码到文件夹下
        try:
            shutil.copy("model_trainer.py", output_folder)
        except Exception as e:
            print("复制文件失败", e)

    def filter_features_by_correlation(self, threshold=0.05):
        """
        根据与标签列的相关度筛选特征，移除相关度绝对值小于阈值的特征。
        返回被移除的特征及其相关度，并保存到日志文件。
        """
        if self.label_col is None or self.labels is None:
            print("未指定标签列，跳过特征筛选。")
            return {}, []

        # 确保标签列是数值类型
        if not pd.api.types.is_numeric_dtype(self.labels):
            print(f"标签列 '{self.label_col}' 不是数值类型，无法计算相关度，跳过特征筛选。")
            return {}, []

        # 合并特征和标签
        df = pd.concat([self.features, self.labels.rename(self.label_col)], axis=1)

        # 筛选数值型特征
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.label_col not in numeric_features:
            print(f"标签列 '{self.label_col}' 不是数值类型，无法计算相关度。")
            return {}, []

        numeric_df = df[numeric_features]

        # 计算相关系数
        corr_matrix = numeric_df.corr()
        label_corr = corr_matrix[self.label_col].abs()
        features_corr = label_corr.drop(self.label_col, errors='ignore').fillna(0)

        # 筛选特征
        selected_features = features_corr[features_corr >= threshold].index.tolist()
        removed_features = features_corr[features_corr < threshold].index.tolist()

        print(f"以下列与标签列相关系数过低，将被删除：{removed_features}")

        # 保存被移除特征信息
        removed_features_info = {feat: features_corr[feat] for feat in removed_features}

        # 更新数据
        self.features = self.features[selected_features]
        self.feature_names = selected_features

        return removed_features_info, selected_features

    def process(self, output_numpy_array=False, output_row_features=False, correlation_threshold=0.05):
        """
        依次执行数据处理的各个步骤。

        :param correlation_threshold:
        :param output_numpy_array: 是否将特征和标签保存为NumPy数组文件。
        :param output_row_features: 是否将每行特征和标签保存为单独的NumPy文件。
        """
        self.read_file()
        self.extract_features_and_labels()
        self.process_mixed_types()
        self.process_categorical_and_date_columns()
        self.replace_nan()
        # 新增特征筛选步骤
        removed_features_info = {}
        if correlation_threshold is not None:
            removed_features_info, _ = self.filter_features_by_correlation(threshold=correlation_threshold)

        self.generate_feature_name_to_index()
        self.generate_feature_name_to_index()
        self.save_features_and_labels(output_numpy_array=output_numpy_array,
                                      output_row_features=output_row_features,
                                      removed_features_info=removed_features_info)


if __name__ == '__main__':
    # 初始化FeatureExtractor实例
    extractor = FeatureExtractor(
        file_path='new_loan_data.xls',  # 可以是.csv, .xlsx, .xls文件
        # exclude_cols=['act_idn_sky', 'cdzbxyfg', 'product_type_new2'],
        label_col='RISK_GRADE',
        # id_col='id_column_name',  # 如果有ID列，指定ID列名字；如果没有，可以不指定
        drop_first_col=False,  # 默认为False，如有需要可以设置为True,
        existing_mapping_path='mapping_dicts.json'  # 指定现有映射文件
    )

    # 执行数据处理流程，并指定输出方式
    extractor.process(output_numpy_array=True, output_row_features=False, correlation_threshold=0.02)
