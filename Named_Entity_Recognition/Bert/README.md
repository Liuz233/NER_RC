## 目录结构
├─data ---- 数据集路径  
│ └─SemEval2010_task8 ---- SemEval2010_task8数据集  
│ ├─given ---- 竞赛所给数据集以及其衍生的文件  
│ ├─prob ---- 处理为“是否为实体标签”的数据集  
│ ├─original ---- 原始数据集  
│ └─prompt ---- 添加了prompt的数据集  
├─model ---- 模型存储路径  
│ ├─bert-base-uncased ---- bert-base-uncased模型  
│ └─deberta-v3-large ---- deberta-v3-large模型  
├─output ---- 预测结果存放路径  
│ ├─new ---- 先序列分类再实体识别结果  
│ └─ensemble ---- 集成学习结果  
├─src ---- 训练loss图存放路径  
│ └─new ---- 先序列分类再实体识别方法  
│ ├─seq_classification  
│ └─entity_extraction  
├─api.py ---- api方法  
├─data_process.py ---- 数据处理  
├─ensemble.py ---- 集成学习  
├─entity_extraction.py ---- 实体抽取  
└─seq_classify.py ---- 序列分类  
