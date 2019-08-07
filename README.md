# 模型测试

DHCP是rule based model，用来训练网络的模型，且训练的都是地主。

每个模型进行了1000轮测试，胜率如下所示。

| 开始训练时刻 | 训练第n次结果 | 对抗random胜率 | 对抗DHCP胜率 |
| --- | --- | --- | --- |
|0802_1836|78500_44|85|37.9|
|0803_0959|8000|83|37.7|
|0804_0912|4500_57|91.5|53.6|
|0804_1045|3500_53|91.1|54.4|
|0804_1423|3700_54|95.9|56.2|
|0804_2022|lord_scratch4000|94.5|54.3|
|0805_1019|2900_54|93.4|55.8|
|0805_1409|lord_4000|—|58.1|
|0806_1906|zero_lord_3000|—|52|
|0806_1906|zero_up+down_3000（农民对抗地主）|—|17.0+21.3|
|0806_1906|zero_lord_4000|—|50.5|
|0806_1906|zero_up+down_4000（农民对抗地主）|—|18.6+19.3|
|0806_1905|zero_lord_7000|—|42.8|
|0806_1905|zero_up+down_7000（农民对抗地主）|—|15.6+18.7|
|0806_1905|zero_lord_13000|—|28.1|
|0806_1905|zero_up+down_13000（农民对抗地主）|—|13.1+16.7|
|0807_1340(调整γ和状态)|lord_2900_54|—|53.1|
|0807_1340(调整γ和状态)|lord_4000|—|55.7|
|0807_1344(调整γ和状态)|zero_up+down_3000（农民对抗地主）|—|20.8+21.8（规则地主：57.4）|
|0807_1344(调整γ和状态)|zero_up+down_4000（农民对抗地主）|—|17.2+25.8（规则地主：57.0）|
|0807_1344(调整γ和状态)|zero_up+down_6000（农民对抗地主）|—|21.9+23.5（规则地主：54.6）|

<!--|0803_0349|2900_48|93.2|53.7|-->
<!--|0803_0349|5300_54（开始过估计）|91.7|49.6|-->
<!--|0803_0349|8000|90.8|47.8|-->
<!--|0803_0349|10000|89.2|42|-->
