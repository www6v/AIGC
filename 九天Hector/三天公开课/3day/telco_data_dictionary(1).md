# telco_data_dictionary：telco_data数据集说明

​	本数据字典记录了telco_data数据集基本情况，telco_data数据集是一个电信用户流失数据集，用于描述电信用户基本情况与目前是否流失状态。

- 数据集本地存储地址

  ​	telco_data数据集存储在本地文件夹中，存储地址在当前代码目录下，文件名为telco_data.csv。

- 数据集字段解释

| Column Name | Description | Value Range | Value Explanation |
|-------------|-------------|-------------|-------------------|
| customerID | 客户ID，user_demographics数据表主键 |              | 由数字和字母组成的 |
| gender | 用户的性别 | Female, Male | Female (女性), Male (男性) |
| SeniorCitizen | 是否为老人 | 0, 1 | 0 (不是), 1 (是) |
| Partner | 用户是否有伴侣 | Yes, No | Yes (有), No (没有) |
| Dependents | 用户经济是否独立，往往用于判断用户是否已经成年 | No, Yes | Yes (有), No (没有) |
| PhoneService | 用户是否有电话服务 | No, Yes | Yes (有), No (没有) |
| MultipleLines | 用户是否开通了多条电话业务 | No phone service, No, Yes | Yes (有多条电话线业务), No (没有多条电话线业务), No phone service (没有电话服务) |
| InternetService | 用户的互联网服务类型 | DSL, Fiber optic, No | DSL (DSL), Fiber optic (光纤), No (没有) |
| OnlineSecurity | 是否开通网络安全服务 | No, Yes, No internet service | Yes（有）、No（无） or No internetservice（没有网路服务） |
| OnlineBackup | 是否开通在线备份服务 | Yes, No, No internet service | Yes（有）、No（无） or No internetservice（没有网路服务） |
| DeviceProtection | 是否开通设备保护服务 | No, Yes, No internet service | Yes（有）、No（无） or No internetservice（没有网路服务） |
| TechSupport | 是否开通技术支持业务 | No, Yes, No internet service | Yes（有）、No（无） or No internetservice（没有网路服务） |
| StreamingTV | 是否开通网络电视 | No, Yes, No internet service | Yes（有）、No（无） or No internetservice（没有网路服务） |
| tenure | 用户入网时间 |  |  |
| Contract | 合同类型 | Month-to-month, One year, Two year | Month-to-month (月付), One year (一年付), Two year (两年付) |
| PaperlessBilling | 是否无纸化账单 | Yes, No | Yes (是), No (否) |
| PaymentMethod | 支付方式 | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Electronic check (电子检查), Mailed check (邮寄支票), Bank transfer (automatic) (银行转账), Credit card (automatic) (信用卡) |
| MonthlyCharges | 月费用 |  | 用户平均每月支付费用 |
| TotalCharges | 总费用 |  | 截至目前用户总消费金额 |
| Churn | 用户是否流失 | No, Yes | Yes (是), No (否) |







