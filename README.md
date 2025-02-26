# AutoFeatCleaner - 自动化特征工程与数据清洗工具

一键处理CSV/XLSX数据的特征提取与噪声清洗工具，专为高效数据预处理设计，助力机器学习流水线快速落地。

## 核心功能

⚙️ **数据处理能力**  
- **智能读取**：自动识别CSV/XLSX格式，支持编码检测
- **噪声清洗**：清除无效数据，缺失值统一填充为-1
- **高级特征工程**  
  - 日期字段自动分解（年/月/日/时）  
  - 分类变量整数映射（保留字典可追溯）  
  - 混合类型列统一转换  
  - 基于相关性的特征筛选

📁 **多样化输出**  
- 多格式导出：CSV表格、NumPy数组、单行特征文件  
- 自解释型元数据：  
  - 特征名-索引对照表  
  - 分类变量编码字典  
  - ID-标签对应关系  
  - 被移除特征日志

🕒 **工程化设计**  
- 时间戳命名输出文件夹防覆盖  
- 自动ID生成与管理  
- 通过类参数灵活配置

## 快速使用

```python
from FeatureExtractor import FeatureExtractor

# 初始化处理器
processor = FeatureExtractor(
    file_path = '你的数据文件.xlsx',
    label_col = '目标列名',
    exclude_cols = ['需要排除的列'],
    correlation_threshold = 0.05  # 特征筛选阈值
)

# 执行处理流程
processor.process(
    output_numpy_array = True,    # 生成numpy数组
    output_row_features = False   # 不生成单行特征文件
)
```


# AutoFeat - Automated Feature Engineering & Data Cleaning Toolkit

A one-stop solution for automated feature extraction, noise cleaning, and dataset preprocessing from CSV/XLSX files. Designed to streamline your data preparation pipeline with minimal configuration.

## Key Features

🔧 **Core Capabilities**  
- **Smart Data Ingestion**: Auto-detect CSV/XLSX formats with encoding detection
- **Noise Filtering**: Remove invalid entries & handle missing values (-1 placeholder)
- **Advanced Feature Engineering**  
  - Automated datetime feature decomposition (year/month/day/hour)  
  - Categorical variable mapping with dictionary persistence  
  - Mixed-type column unification  
  - Correlation-based feature selection

📊 **Output Flexibility**  
- Multiple export formats: CSV, NumPy arrays, per-row feature files  
- Self-documenting outputs:  
  - Feature name-index mapping  
  - Categorical encoding dictionaries  
  - ID-label relationships  
  - Removed features log

⏱️ **Production-Ready**  
- Time-stamped output folders prevent overwrites  
- Automatic ID generation/management  
- Configurable via simple class parameters

## Usage Example

```python
from FeatureExtractor import FeatureExtractor

processor = FeatureExtractor(
    file_path = 'your_data.xlsx',
    label_col = 'target_column',
    exclude_cols = ['irrelevant_field'],
    correlation_threshold = 0.05
)

processor.process(
    output_numpy_array = True,
    output_row_features = False
)
```
