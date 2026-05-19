"""
智能结构推荐模块。

根据用户设计需求关键词匹配场景，从知识库中推荐结构方案，
支持结合计算结果（应力/载荷水平）进行排序优化。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 结构方案知识库
# ---------------------------------------------------------------------------
_KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "轴系设计": {
        "scenario_keywords": [
            "轴", "轴系", "传动轴", "主轴", "转轴", "轴设计",
            "轴颈", "轴肩", "轴径", "长轴", "短轴", "高速轴",
        ],
        "schemes": [
            {
                "name": "阶梯轴",
                "description": "沿轴线方向直径分段变化的实心轴，是最广泛应用的轴结构形式。通过轴肩实现轴上零件（齿轮、轴承等）的轴向定位。",
                "advantages": [
                    "加工简单，车削即可完成",
                    "轴肩提供可靠的轴向定位基准",
                    "各轴段可根据受力独立优化直径",
                    "成本低，适用于大多数通用机械",
                ],
                "disadvantages": [
                    "轴肩处存在应力集中，需设过渡圆角",
                    "重量较大，不适合高速轻量化场景",
                    "多阶梯导致毛坯余量大，材料利用率一般",
                ],
                "typical_params": {
                    "轴肩高度": "2~5 mm（定位用）；1~2 mm（非定位用）",
                    "过渡圆角半径 r": "≥ 0.05d（d 为轴径）；推荐 ≥ 1 mm",
                    "长径比 L/d": "≤ 15（刚度控制）；≤ 30（强度控制）",
                    "直径公差": "IT6~IT7（轴承位）；IT7~IT8（齿轮位）",
                    "表面粗糙度 Ra": "0.8~1.6 μm（轴承位）；1.6~3.2 μm（一般段）",
                },
                "related_standards": [
                    "GB/T 1569 圆柱形轴伸",
                    "GB/T 1095 平键键槽",
                    "GB/T 275 滚动轴承配合",
                    "JB/T 7518 阶梯轴设计规范",
                ],
            },
            {
                "name": "空心轴",
                "description": "中心有通孔或盲孔的轴结构，在相同重量下具有更大的抗扭/抗弯截面模量，适用于需要减重或内部走线/通油的场合。",
                "advantages": [
                    "重量轻（相同外径下比实心轴轻 20%~50%）",
                    "抗扭截面系数利用率高（材料分布远离中性轴）",
                    "可内穿冷却油/电线/拉杆",
                    "适用于航空、赛车等轻量化场景",
                ],
                "disadvantages": [
                    "制造工艺复杂（需深孔钻或热轧无缝管）",
                    "内孔表面质量难保证，可能成为疲劳裂纹源",
                    "径向刚度略低于同外径实心轴",
                    "成本约为同规格阶梯轴的 1.5~3 倍",
                ],
                "typical_params": {
                    "内外径比 di/do": "0.5~0.75（推荐范围）",
                    "壁厚": "≥ 3 mm（考虑加工刚度）",
                    "内孔同轴度": "≤ 0.02 mm（高转速）",
                    "内外表面粗糙度": "Ra 0.8~1.6 μm",
                },
                "related_standards": [
                    "GB/T 8162 结构用无缝钢管",
                    "GB/T 3639 冷拔精密无缝钢管",
                    "GB/T 3094 冷拔异型钢管",
                ],
            },
            {
                "name": "组合轴",
                "description": "由两个或多个轴段通过联轴器、焊接或法兰连接而成的轴系，用于特长轴或需要分段装配/维修的场合。",
                "advantages": [
                    "可分段制造和运输，降低单段加工难度",
                    "便于局部维修（只需更换损坏段）",
                    "可通过联轴器补偿安装误差和热膨胀",
                    "适用于跨距极长（>10m）的传动系统",
                ],
                "disadvantages": [
                    "联轴器处增加重量和成本",
                    "联轴器处存在附加不平衡量和间隙",
                    "对中精度要求高，安装调试复杂",
                    "联接部位的疲劳强度低于整体轴",
                ],
                "typical_params": {
                    "联接方式": "凸缘联轴器 / 膜片联轴器 / 万向联轴器",
                    "对中精度": "≤ 0.05 mm（刚性联轴器）；≤ 0.1 mm（弹性联轴器）",
                    "单段长度": "≤ 6 m（一般加工能力上限）",
                    "联接强度校核": "按螺栓/焊缝承载能力评定",
                },
                "related_standards": [
                    "GB/T 5843 凸缘联轴器",
                    "GB/T 5272 膜片联轴器",
                    "GB/T 4323 弹性套柱销联轴器",
                ],
            },
        ],
    },
    "齿轮传动": {
        "scenario_keywords": [
            "齿轮", "齿轮传动", "减速器", "增速器", "齿传动",
            "传动比", "啮合", "齿轮箱", "齿轮副",
        ],
        "schemes": [
            {
                "name": "直齿圆柱齿轮",
                "description": "齿线平行于轴线的圆柱齿轮，是最基本的齿轮形式。适用于平行轴之间的中等载荷和中低速传动。",
                "advantages": [
                    "制造简单，成本最低",
                    "无轴向力，轴承选型简单",
                    "传动效率高（单级 η≥98%）",
                    "安装调试方便，对轴向位置不敏感",
                ],
                "disadvantages": [
                    "重合度较小（通常 1.2~1.8），传动平稳性一般",
                    "高速时噪声和振动较大",
                    "齿根弯曲强度受限于齿数较少时的根切",
                    "不适合极高转速（v>25 m/s）",
                ],
                "typical_params": {
                    "模数 m": "1.5~20 mm（一般机械）；0.5~1.25 mm（仪表）",
                    "齿数 z1": "≥ 17（标准齿，避免根切）；≥ 14（变位齿）",
                    "齿宽系数 φ_d": "0.6~1.2（一般）；0.3~0.6（悬臂布置）",
                    "传动比 i（单级）": "≤ 6（一般）；≤ 10（开式）",
                    "精度等级": "GB/T 10095 7~9 级（一般）；5~6 级（精密）",
                },
                "related_standards": [
                    "GB/T 1356 基本齿廓",
                    "GB/T 10095.1 精度制",
                    "GB/T 3480 承载能力计算",
                ],
            },
            {
                "name": "斜齿圆柱齿轮",
                "description": "齿线与轴线成一定螺旋角的圆柱齿轮，啮合时逐齿进入接触，运转平稳。适用于高速重载平行轴传动。",
                "advantages": [
                    "重合度大（可达 3~5），传动平稳、噪声低",
                    "齿面接触线倾斜，承载能力比同规格直齿提高 20%~40%",
                    "适用于高速传动（v=30~100 m/s）",
                    "最小齿数可降低至 12~14（不根切）",
                ],
                "disadvantages": [
                    "产生轴向力，需配推力轴承或成对（左右旋）抵消",
                    "加工需专用滚齿机（差动机构），成本较高",
                    "齿面滑动较大，发热和效率略低于直齿",
                    "对装配精度要求较高",
                ],
                "typical_params": {
                    "螺旋角 β": "8°~20°（一般）；20°~35°（大螺旋角）",
                    "模数 m_n": "1~16 mm",
                    "当量齿数 zv": "z / cos³β",
                    "轴向力 Fa": "Ft × tanβ",
                    "推荐成对使用": "左右旋各一，抵消轴向力",
                },
                "related_standards": [
                    "GB/T 1356 基本齿廓",
                    "GB/T 10095.1 精度制",
                    "GB/T 3480 承载能力计算",
                    "ISO 6336 齿轮强度计算",
                ],
            },
            {
                "name": "锥齿轮（伞齿轮）",
                "description": "分度曲面为圆锥面的齿轮，用于相交轴（通常 90°）之间的运动和动力传递。",
                "advantages": [
                    "实现相交轴传动（典型 90°），改变传动方向",
                    "承载能力优于蜗轮蜗杆",
                    "传动效率较高（直齿锥 η≈95%；弧齿锥 η≈97%）",
                    "弧齿锥齿轮运转平稳，适合高速",
                ],
                "disadvantages": [
                    "加工复杂，需专用刨齿/铣齿机床",
                    "对安装精度敏感（锥顶必须重合）",
                    "轴向力较大，支撑结构复杂",
                    "齿面接触区需配对研磨保证",
                ],
                "typical_params": {
                    "大端模数 m": "1.5~16 mm",
                    "锥距 R": "决定齿轮副尺寸",
                    "齿宽 b": "≤ min(R/3, 10m)",
                    "传动比 i（单级）": "≤ 4（一般）；≤ 6（弧齿锥）",
                    "轴交角 Σ": "90°（最常用）",
                },
                "related_standards": [
                    "GB/T 12369 锥齿轮图样",
                    "GB/T 11365 锥齿轮精度制",
                    "GB/T 10062 锥齿轮承载能力",
                ],
            },
            {
                "name": "蜗轮蜗杆传动",
                "description": "由蜗杆和蜗轮组成，用于空间交错轴（通常 90°）之间的传动，可实现大传动比和自锁功能。",
                "advantages": [
                    "单级传动比大（i=5~80），结构紧凑",
                    "传动平稳、噪声极低（滑动接触）",
                    "可实现自锁（导程角 γ < 当量摩擦角 ρ'）",
                    "适用于分度机构、升降机等精密定位",
                ],
                "disadvantages": [
                    "传动效率低（η=50%~90%，自锁时<50%）",
                    "齿面滑动速度大，发热严重，需良好润滑和散热",
                    "蜗轮需用贵重的减摩材料（青铜等）",
                    "承载能力受齿面胶合极限限制",
                ],
                "typical_params": {
                    "模数 m": "1~20 mm",
                    "蜗杆头数 z1": "1~4（自锁取 1；高效取 4~6）",
                    "导程角 γ": "3°~25°（γ < 3°35' 自锁）",
                    "中心距 a": "根据功率和速比查表",
                    "散热面积校核": "P × (1-η) ≤ Kt × A × Δt",
                },
                "related_standards": [
                    "GB/T 10085 圆柱蜗杆基本参数",
                    "GB/T 10089 蜗杆精度制",
                    "GB/T 3944 蜗杆传动承载能力",
                ],
            },
        ],
    },
    "联接方式": {
        "scenario_keywords": [
            "联接", "连接", "键", "花键", "过盈", "螺纹联接",
            "紧固", "装配", "可拆", "不可拆", "联轴器",
        ],
        "schemes": [
            {
                "name": "平键联接",
                "description": "利用键的侧面传递扭矩的最常用轴毂联接方式。键的上下面与键槽有间隙，依靠侧面挤压传递扭矩。",
                "advantages": [
                    "结构简单，标准化程度极高",
                    "装拆方便，可重复使用",
                    "成本极低（标准件采购）",
                    "对中性较好（双键或花键更优）",
                ],
                "disadvantages": [
                    "键槽削弱轴和轮毂强度（应力集中系数 Kt≈1.6~2.0）",
                    "单键承载能力有限，重载需双键或花键",
                    "不能承受轴向力（仅导向平键可轴向滑动）",
                    "存在反向间隙，不适合频繁正反转",
                ],
                "typical_params": {
                    "键宽 b": "按轴径 d 查表（GB/T 1096）",
                    "键高 h": "通常 = b",
                    "键长 L": "(1.5~2.5)d，≤ 轮毂宽度 - 2mm",
                    "键槽深度 t": "轴上: ~0.6h；轮毂: ~0.4h",
                    "许用挤压应力 [σ_p]": "60~150 MPa（静联接，视材料）",
                },
                "related_standards": [
                    "GB/T 1096 普通平键",
                    "GB/T 1095 键槽剖面",
                    "GB/T 1567 薄型平键",
                ],
            },
            {
                "name": "花键联接",
                "description": "轴与轮毂上加工出多个均布的齿形突起的联接方式。分为矩形花键和渐开线花键，传递扭矩远大于平键。",
                "advantages": [
                    "承载能力大（多齿同时受力，约为平键的3~6倍）",
                    "对中性好（间隙或过渡配合）",
                    "轴上零件可沿花键轴向滑动（如变速箱齿轮）",
                    "应力集中较平键均匀，疲劳强度高",
                ],
                "disadvantages": [
                    "加工成本高（需拉床/滚齿机/花键磨床）",
                    "标准型号多但不如平键通用",
                    "轴向滑动型的磨损和微动腐蚀风险",
                    "花键轴的弯曲疲劳缺口系数较大",
                ],
                "typical_params": {
                    "矩形花键规格": "N×d×D×B（键数×小径×大径×键宽）",
                    "渐开线花键": "模数制 m=0.5~10 mm，压力角 30°/37.5°/45°",
                    "键数": "矩形: 6/8/10；渐开线: z=10~60",
                    "许用挤压应力 [σ_p]": "80~200 MPa（静载荷，调质钢）",
                },
                "related_standards": [
                    "GB/T 1144 矩形花键",
                    "GB/T 3478.1 渐开线花键",
                    "DIN 5480 渐开线花键",
                ],
            },
            {
                "name": "过盈配合联接",
                "description": "通过轴与孔的过盈量产生的径向压力来传递扭矩和/或轴向力，属于不可拆（或难拆）联接。",
                "advantages": [
                    "无键槽，不削弱轴截面，疲劳强度高",
                    "对中性极好（配合面全周接触）",
                    "可同时传递扭矩和轴向力",
                    "结构紧凑，无需额外零件",
                ],
                "disadvantages": [
                    "配合面加工精度要求高（IT5~IT7）",
                    "装拆困难（需加热/液压/压入），不利于维修",
                    "微动磨损和微动疲劳风险（尤其在振动环境）",
                    "过盈量对温度变化敏感（热膨胀差可能松动）",
                ],
                "typical_params": {
                    "过盈量 δ": "根据传递扭矩计算，通常 0.001d~0.002d",
                    "配合": "H7/s6、H7/u6、H7/x6 等",
                    "表面粗糙度 Ra": "≤ 0.8 μm",
                    "压入应力": "注意轮毂拉应力不超过屈服/脆断极限",
                    "温差装配": "加热温差 ≥ 100°C（轮毂）；液氮冷却（轴）",
                },
                "related_standards": [
                    "GB/T 5371 过盈配合计算",
                    "GB/T 1800 公差与配合",
                    "DIN 7190 过盈配合设计",
                ],
            },
            {
                "name": "螺纹联接",
                "description": "通过螺栓/螺钉/螺柱的预紧力产生摩擦力或剪切力传递载荷，是最通用的可拆联接方式。",
                "advantages": [
                    "标准化程度极高，采购方便",
                    "装拆方便，可重复使用（注意预紧力衰减）",
                    "适应性强（不同规格/材料/防松方式）",
                    "可施加可控预紧力（力矩扳手/液压张紧器）",
                ],
                "disadvantages": [
                    "预紧力分散性大（摩擦力系数分散 20%~30%）",
                    "振动环境下易松脱，需防松措施",
                    "螺纹根部应力集中严重（Kt 可达 3~5）",
                    "多螺栓组载荷分配不均（受被联接件刚度影响）",
                ],
                "typical_params": {
                    "预紧系数": "0.5~0.7（一般）；0.7~0.85（重要）",
                    "螺栓间距 t": "≤ 7d（密封）; ≤ 10d（一般）",
                    "防松方式": "弹簧垫圈/尼龙嵌件/螺纹胶/双螺母/机械锁紧",
                    "强度等级": "4.8/8.8/10.9/12.9（GB/T 3098.1）",
                },
                "related_standards": [
                    "GB/T 3098.1 螺栓机械性能",
                    "GB/T 16823.1 螺栓应力面积",
                    "VDI 2230 高强度螺栓系统计算",
                ],
            },
        ],
    },
    "轴承选型": {
        "scenario_keywords": [
            "轴承", "滚动轴承", "滑动轴承", "径向轴承", "推力轴承",
            "轴承选型", "轴承配置", "支撑方式",
        ],
        "schemes": [
            {
                "name": "深沟球轴承",
                "description": "最广泛使用的滚动轴承，主要承受径向载荷，也可承受一定的双向轴向载荷。摩擦系数低，极限转速高。",
                "advantages": [
                    "标准化程度最高，价格最低",
                    "极限转速高（适合高速轻载）",
                    "可同时承受径向和轴向载荷",
                    "密封/防尘型可选，免维护周期长",
                    "噪音低，适合精密设备",
                ],
                "disadvantages": [
                    "承载能力在球轴承中属于中等",
                    "轴向承载能力有限（通常 ≤0.7Fr）",
                    "不耐冲击载荷",
                    "安装对中要求较高（允许偏转角 ≤0.05°）",
                ],
                "typical_params": {
                    "内径范围": "3~1320 mm",
                    "极限转速系数 dn": "≤ 1,000,000（脂润滑）；≤ 1,500,000（油润滑）",
                    "额定寿命 L10": "按 ISO 281 计算，一般机械 8000~20000 h",
                    "游隙组": "C2/CN/C3/C4（CN 为默认）",
                    "密封形式": "ZZ（铁盖）/ 2RS（橡胶密封）/ 开式",
                },
                "related_standards": [
                    "GB/T 276 深沟球轴承",
                    "GB/T 6391 额定动载荷",
                    "ISO 15 外形尺寸",
                ],
            },
            {
                "name": "圆柱滚子轴承",
                "description": "滚子与滚道为线接触，径向承载能力远大于同尺寸球轴承。适用于重载或冲击载荷工况。",
                "advantages": [
                    "径向承载能力为同尺寸深沟球的 1.5~3 倍",
                    "刚性高，适用于精密主轴支撑",
                    "极限转速较高（仅次于球轴承）",
                    "可分离设计便于装拆（内外圈可分别安装）",
                    "耐冲击载荷性能好",
                ],
                "disadvantages": [
                    "不能承受轴向载荷（N/NU 型）或仅能承受单向轴向力（NJ/NF 型）",
                    "对偏斜和不对中敏感（允许偏转角 ≤0.02°）",
                    "噪声高于球轴承",
                    "价格比深沟球高 50%~150%",
                ],
                "typical_params": {
                    "内径范围": "10~2000 mm",
                    "系列": "NU（外圈可分离）/ N（内圈可分离）/ NJ / NUP",
                    "适用于": "Fr/Fa > 4 的场合（径向为主）",
                    "保持架": "车制黄铜（重载）/ 冲压钢（一般）",
                },
                "related_standards": [
                    "GB/T 283 圆柱滚子轴承",
                    "GB/T 6391 额定动载荷",
                    "ISO 246 圆柱滚子轴承",
                ],
            },
            {
                "name": "角接触球轴承",
                "description": "可同时承受径向和轴向联合载荷，接触角越大轴向承载能力越强。通常成对使用（背对背/面对面/串联）。",
                "advantages": [
                    "可承受径向+轴向联合载荷",
                    "精度高，适用于主轴、丝杠支撑",
                    "转速高（介于深沟球和圆柱滚子之间）",
                    "可通过预紧提高刚度和旋转精度",
                    "成对配置灵活（DB/DF/DT 三种布置）",
                ],
                "disadvantages": [
                    "只能承受单向轴向力（单列），需成对使用",
                    "对安装方向和预紧量有严格要求",
                    "价格高于深沟球",
                    "润滑要求较严格（预紧发热需控制）",
                ],
                "typical_params": {
                    "接触角 α": "15°（高速精密切削）/ 25°（一般）/ 40°（重载轴向）",
                    "成对配置": "DB（背对背，抗弯矩）/ DF（面对面）/ DT（串联）",
                    "预紧方式": "定位预紧（隔圈）/ 定压预紧（弹簧）",
                    "典型应用": "机床主轴 / 滚珠丝杠支撑 / 齿轮箱",
                },
                "related_standards": [
                    "GB/T 292 角接触球轴承",
                    "GB/T 6391 额定动载荷",
                    "ISO 12044 角接触球轴承",
                ],
            },
            {
                "name": "推力轴承",
                "description": "专门承受轴向载荷的轴承。分为推力球轴承（高速轻载）和推力滚子轴承（重载）。",
                "advantages": [
                    "轴向承载能力极强（推力滚子轴承可达数万kN）",
                    "可自调心（推力调心滚子轴承）",
                    "刚度高",
                    "适用于垂直轴支撑、回转工作台等",
                ],
                "disadvantages": [
                    "不能承受径向载荷（需配合径向轴承使用）",
                    "极限转速低（离心力使滚球偏离滚道）",
                    "推力球轴承不允许有偏斜",
                    "润滑要求较高（滑动速度大）",
                ],
                "typical_params": {
                    "类型": "推力球（511/512/513/514 系列）/ 推力圆柱滚子 / 推力调心滚子",
                    "极限转速": "推力球 dn ≤ 500,000；推力滚子 dn ≤ 300,000",
                    "最小轴向载荷": "约 0.01C（防止滑动损伤）",
                    "安装": "座圈与壳体间隙配合；轴圈与轴过盈配合",
                },
                "related_standards": [
                    "GB/T 301 推力球轴承",
                    "GB/T 5859 推力调心滚子轴承",
                    "GB/T 6391 额定动载荷",
                ],
            },
        ],
    },
    "密封方式": {
        "scenario_keywords": [
            "密封", "防漏", "油封", "密封件", "轴封",
            "密封装置", "防尘", "防水", "气密",
        ],
        "schemes": [
            {
                "name": "唇形密封（骨架油封）",
                "description": "由弹性橡胶唇口与金属骨架组成的接触式旋转密封件。唇口在弹簧和介质压力下紧贴旋转轴表面形成密封。",
                "advantages": [
                    "标准化程度最高（GB/T 9877），采购方便",
                    "安装简单，成本极低（几元到几十元）",
                    "密封效果好（线速度 ≤15 m/s 时泄漏率极低）",
                    "内/外密封、防尘/防油通用",
                ],
                "disadvantages": [
                    "唇口与轴摩擦发热，有功率损耗",
                    "高速时（v>20 m/s）唇口烧焦失效",
                    "轴表面粗糙度和硬度要求较高（Ra≤0.4μm, HRC≥45）",
                    "使用寿命有限（一般 2000~8000 h），需定期更换",
                ],
                "typical_params": {
                    "适用线速度": "≤ 15 m/s（NBR）；≤ 25 m/s（FKM）",
                    "适用温度": "-30~100°C（NBR）；-20~200°C（FKM）",
                    "轴表面硬度": "≥ HRC 45（磨损保护）",
                    "轴表面粗糙度 Ra": "0.2~0.4 μm",
                    "唇口过盈量": "0.5~1.5 mm（按轴径）",
                },
                "related_standards": [
                    "GB/T 9877 旋转轴唇形密封圈",
                    "GB/T 13871 密封元件",
                    "ISO 6194 旋转轴唇形密封",
                ],
            },
            {
                "name": "迷宫密封",
                "description": "由一系列节流间隙和膨胀腔组成的非接触式密封，通过流体在迷宫路径中的多次节流膨胀来限制泄漏。",
                "advantages": [
                    "无接触、无磨损、无摩擦热",
                    "适用于极高转速（v>100 m/s）",
                    "寿命极长（几乎免维护）",
                    "可适应高温工况（取决于材料）",
                ],
                "disadvantages": [
                    "不能完全密封（存在一定泄漏量）",
                    "轴向空间占用较大",
                    "制造精度要求高（间隙需精确控制）",
                    "静止时无密封效果（依赖轴的旋转泵送效应）",
                ],
                "typical_params": {
                    "节流间隙": "0.2~0.5 mm（一般）；0.05~0.15 mm（精密）",
                    "迷宫齿数": "3~6 齿（一般）；6~12 齿（高密封要求）",
                    "腔深与间隙比": "≥ 3:1（保证膨胀效果）",
                    "常用材料": "铝/铜合金（轻载）；不锈钢（腐蚀介质）",
                    "组合密封": "迷宫+甩油环+回油槽（轴承座密封常用）",
                },
                "related_standards": [
                    "JB/T 8725 迷宫密封型式",
                    "API 610 离心泵密封",
                ],
            },
            {
                "name": "机械密封（端面密封）",
                "description": "由垂直于轴线的两个精密端面（一动一静）贴合形成密封。通过弹簧和介质压力维持端面贴合力，是目前最可靠的旋转密封方式。",
                "advantages": [
                    "泄漏率极低（<1 mL/h，满足环保要求）",
                    "寿命长（正常工况 2~5 年）",
                    "功率消耗低（端面液膜润滑）",
                    "适用于高压（≤45 MPa）、高温（≤400°C）、高速",
                ],
                "disadvantages": [
                    "结构复杂，成本高（数百到数万元）",
                    "安装精度要求高（端面跳动 ≤0.002 mm）",
                    "需要冲洗/冷却辅助系统",
                    "对介质洁净度敏感（颗粒导致端面磨损）",
                ],
                "typical_params": {
                    "端面材料": "碳石墨 / SiC / WC（配对组合）",
                    "弹簧形式": "单弹簧 / 多弹簧 / 波纹管",
                    "适用压力": "≤ 4.5 MPa（非平衡型）；≤ 45 MPa（平衡型）",
                    "适用温度": "-50~400°C（取决于副密封材料）",
                    "泄漏率": "< 1~5 mL/h（液体）",
                },
                "related_standards": [
                    "GB/T 14211 机械密封试验方法",
                    "GB/T 33509 机械密封通用规范",
                    "API 682 泵用机械密封",
                    "ISO 21049 机械密封",
                ],
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# 场景关键词权重映射：推荐类关键词 → 场景名称
# ---------------------------------------------------------------------------
_SCENE_MAP: dict[str, str] = {}
for _scene_name, _scene_data in _KNOWLEDGE_BASE.items():
    for _kw in _scene_data["scenario_keywords"]:
        _SCENE_MAP[_kw] = _scene_name

# 场景专属鉴别关键词：命中即对该场景施加高额加分，避免短词（如"轴"）的泛匹配
_DISCRIMINATOR_BONUS = 100
_SCENE_DISCRIMINATORS: dict[str, tuple[str, ...]] = {
    "密封方式": ("密封", "防漏", "油封", "密封件", "防尘", "防水", "气密", "迷宫密封"),
    "轴承选型": ("轴承", "滚动体", "保持架", "游隙", "深沟球", "角接触", "圆柱滚子"),
    "联接方式": ("键联接", "花键", "过盈配合", "平键", "半圆键", "螺纹联接"),
    "齿轮传动": ("齿轮", "蜗轮", "蜗杆", "齿面", "啮合", "减速器", "增速器"),
    "轴系设计": ("阶梯轴", "空心轴", "组合轴", "轴系", "长轴", "短轴"),
}


def _calc_torque_level(calc_result: dict | None) -> str | None:
    """从计算结果中提取扭矩水平标签。"""
    if not calc_result:
        return None
    T = calc_result.get("torque_Nm")
    if T is None:
        return None
    if T > 5000:
        return "heavy_torque"
    if T > 500:
        return "medium_torque"
    return "light_torque"


def _calc_stress_level(calc_result: dict | None) -> str | None:
    """从计算结果中提取应力水平标签。"""
    if not calc_result:
        return None
    stress = calc_result.get("max_shear_stress_MPa") or calc_result.get("max_bending_stress_MPa")
    if stress is None:
        return None
    if stress > 200:
        return "heavy_stress"
    if stress > 50:
        return "medium_stress"
    return "light_stress"


def _score_scheme(query_lower: str, scene_name: str, scheme: dict, calc_result: dict | None) -> float:
    """计算方案与查询的匹配得分。"""
    score = 0.0

    # 关键词命中得分（在方案 name/description 中匹配 query 中的词）
    query_words = set(query_lower.replace("、", " ").replace("，", " ").split())
    name_hits = sum(1 for w in query_words if w in scheme["name"].lower())
    desc_hits = sum(0.5 for w in query_words if w in scheme["description"].lower())
    score += name_hits * 5.0 + desc_hits * 1.5

    # 场景匹配基础分
    if scene_name in query_lower:
        score += 3.0

    # 结合计算结果的载荷水平调整
    if calc_result:
        torque_lvl = _calc_torque_level(calc_result)
        stress_lvl = _calc_stress_level(calc_result)

        # 高载荷 → 提升重载方案评分
        if torque_lvl == "heavy_torque" or stress_lvl == "heavy_stress":
            heavy_bonus_keywords = ["花键", "过盈", "圆柱滚子", "推力", "机械密封", "斜齿", "蜗轮"]
            if any(kw in scheme["name"] for kw in heavy_bonus_keywords):
                score += 4.0
        # 低载荷 → 提升经济方案评分
        elif torque_lvl == "light_torque" and stress_lvl == "light_stress":
            light_bonus_keywords = ["平键", "深沟球", "唇形", "直齿"]
            if any(kw in scheme["name"] for kw in light_bonus_keywords):
                score += 3.0

    return score


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
def recommend_structure(query: str, calc_result: dict | None = None) -> list[dict[str, Any]]:
    """根据用户查询关键词匹配场景并推荐结构方案。

    Args:
        query: 用户设计需求描述
        calc_result: 可选，来自 calculator.py 的计算结果，用于按应力/载荷水平排序

    Returns:
        推荐方案列表（最多 3 个），每个元素包含方案详情 + match_score
        若无法匹配场景则返回空列表
    """
    q_lower = (query or "").strip().lower()
    if not q_lower:
        return []

    # 确定场景：在 query 中匹配场景关键词（长关键词权重更高）
    # 先检查场景专属鉴别关键词（命中即大额加分）
    scene_scores: dict[str, float] = {}
    for scene_name, disc_kws in _SCENE_DISCRIMINATORS.items():
        for dkw in disc_kws:
            if dkw in q_lower:
                scene_scores[scene_name] = scene_scores.get(scene_name, 0) + _DISCRIMINATOR_BONUS
                break  # 一个鉴别词命中即足够

    # 通用关键词匹配（权重 = 长度平方，长词 > 短词）
    for kw, scene_name in _SCENE_MAP.items():
        if kw in q_lower:
            weight = len(kw) * len(kw)
            scene_scores[scene_name] = scene_scores.get(scene_name, 0) + weight

    if not scene_scores:
        return []

    # 取命中关键词最多的场景
    best_scene = max(scene_scores, key=lambda k: scene_scores[k])
    scene_data = _KNOWLEDGE_BASE[best_scene]
    schemes = scene_data["schemes"]

    # 为每个方案打分并排序
    scored: list[tuple[dict[str, Any], float]] = []
    for scheme in schemes:
        s = _score_scheme(q_lower, best_scene, scheme, calc_result)
        scored.append((scheme, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 构建结果
    results: list[dict[str, Any]] = []
    for scheme, score in scored[:3]:
        results.append({
            "name": scheme["name"],
            "description": scheme["description"],
            "advantages": scheme["advantages"],
            "disadvantages": scheme["disadvantages"],
            "typical_params": scheme["typical_params"],
            "related_standards": scheme["related_standards"],
            "match_score": round(score, 1),
            "scene": best_scene,
        })

    return results


def format_recommendation(recommendations: list[dict[str, Any]]) -> str:
    """将推荐结果格式化为结构化文本。

    Args:
        recommendations: recommend_structure() 返回的列表

    Returns:
        格式化的多行文本字符串
    """
    if not recommendations:
        return "（未匹配到适用场景，建议补充更多设计约束信息后重试）"

    lines: list[str] = []
    scene = recommendations[0].get("scene", "未知场景")
    lines.append(f"匹配场景: {scene}")
    lines.append(f"共匹配 {len(recommendations)} 个推荐方案\n")

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"┌─ 推荐方案 {i}: {rec['name']}（匹配度: {rec['match_score']}）")
        lines.append(f"│  适用场景: {rec['description']}")
        lines.append(f"│  优点:")
        for adv in rec["advantages"]:
            lines.append(f"│    + {adv}")
        lines.append(f"│  缺点:")
        for dis in rec["disadvantages"]:
            lines.append(f"│    - {dis}")
        lines.append(f"│  典型参数范围:")
        for param, val in rec["typical_params"].items():
            lines.append(f"│    · {param}: {val}")
        lines.append(f"│  相关标准:")
        for std in rec["related_standards"]:
            lines.append(f"│    · {std}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 测试块
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("智能结构推荐模块 — 单元测试")
    print("=" * 60)

    # 测试1: 轴系设计推荐
    print("\n[1] 轴系设计推荐")
    recs = recommend_structure("我需要设计一根高速传动轴，选择什么结构形式比较好")
    print(format_recommendation(recs))

    # 测试2: 齿轮传动推荐 + 结合计算结果
    print("\n[2] 齿轮传动推荐（结合重载计算结果）")
    calc = {"torque_Nm": 8000, "max_shear_stress_MPa": 250}
    recs = recommend_structure("减速器齿轮选型，传递大扭矩", calc_result=calc)
    print(format_recommendation(recs))

    # 测试3: 轴承选型
    print("\n[3] 轴承选型推荐")
    recs = recommend_structure("机床主轴用什么轴承合适")
    print(format_recommendation(recs))

    # 测试4: 联接方式推荐
    print("\n[4] 联接方式推荐")
    recs = recommend_structure("重载轴毂联接怎么选")
    print(format_recommendation(recs))

    # 测试5: 密封方式
    print("\n[5] 密封方式推荐")
    recs = recommend_structure("高速旋转轴的密封方案")
    print(format_recommendation(recs))

    # 测试6: 无法匹配
    print("\n[6] 无法匹配场景")
    recs = recommend_structure("今天天气怎么样")
    print(f"  返回空列表: {recs == []} (长度={len(recs)})")

    print("\n" + "=" * 60)
    print("所有测试完成，无异常报错。")
    print("=" * 60)
