# Modularization
**# Modularization**



文件组织结构：

```
--decomposeDNNintoModules 
    --datasets				        用于保存模型训练、测试数据（最后没用上）
    --models          		        保存训练好的模型（待模块化模型），命名规则为"训练集名称"+"_"+"隐藏层数量"
    --modularized_ models	        保存模块化后模型。
    --result          		        用于保存分析结果
    --modularize_ti.py		        按照decomposeDNNintoModules中的方法1TI，使用CI和TI方法模块化。
    --reuse.py 				        对应decomposeDNNintoModules中的方法1TIreuse.py文件（源代码跑的太慢了，进行了修改）
    --evaluate.py			        文件包含对模块化后的模型准确率测试和相似性测试。
    --concern_identification.py     实现CI
    --concern_modularzation.py      实现CM（没写完）
    --tangling_identification.py    实现TI
    
```

decomposeDNNintoModules中参照decomposeDNNintoModules库(https://github.com/rangeetpan/decomposeDNNintoModules/tree/master) 复现论文《On decomposing a deep neural network into modules》模块化方法

对模型进行模块运行modularize_ti.py文件即可，reuse.py是重用文件，evaluate.py对模块化后模型进行评估


