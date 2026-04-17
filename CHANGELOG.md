# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **Lakehouse Lite — 全量学术数据架构** (Parquet + DuckDB + SQLite FTS5)
  - `extract_to_parquet.py` — 从583GB OpenAlex snapshot提取全部2.97亿篇论文到Parquet
    - 年份分区: `works_YYYY.parquet` + `authorships_YYYY.parquet`
    - 20列paper schema + 7列authorship schema
    - 流式写入 (PyArrow ParquetWriter), 50K rows/batch, 内存可控
    - `--test N` 小规模测试, `--resume` 断点续传, `--verify` 验证输出
  - `build_derived.py` — 从Parquet构建派生查询层 (每个可独立重建)
    - (A) DuckDB `analytics.duckdb` — 全量paper表+索引, authorships视图直读Parquet
    - (B) SQLite `search.db` — FTS5全文检索 (title+abstract), 列名兼容旧schema
    - (C) `authors.parquet` — 从authorships用DuckDB GROUP BY去重
  - `analyze.py` 自动检测: 有search.db用lakehouse, 否则fallback到legacy openalex.db
  - 数据存储: N:/academic-data/parquet/ (源) + N:/academic-data/derived/ (派生)
  - 新依赖: `duckdb` (pyarrow已有)

### Changed

- **Paper Landscape: datamapplot 替代 Plotly** — DeckGL/WebGL 语义缩放地图
  - 用 `datamapplot.create_interactive_plot()` 生成交互式HTML, 通过iframe嵌入Dash
  - **三层语义缩放**: 缩小看5个field大标签 → 放大出现~42个subfield → 继续放大显示405个cluster_label
  - **内置搜索**: datamapplot原生搜索框, 输入关键词高亮匹配论文
  - **Topic树**: `enable_topic_tree=True`, 侧边栏显示标签层级结构
  - **Cluster边界线**: `cluster_boundary_polygons=True`, 清晰区分领域边界
  - Hover显示论文标题+年份+期刊+citation数+tier
  - DeckGL引擎25K论文流畅渲染, HTML文件2.5MB, 生成~5s
  - 时间演化控件隐藏 (datamapplot的搜索+缩放已足够强大)
  - 原Plotly `build_scatter()`/`build_territory_scatter()` 保留作为fallback
  - Author Landscape tab完全不受影响
  - 右侧面板(Paper Details / AI Cluster Analysis / AI Landscape)保留

- **Paper Landscape 地图学风格重构 v2** — 层级领地 + 标签分布优化
  - **5个field领地** (Business/Social Sci/CS/Decision Sci/Other), 从8个subfield→5个field
  - **预渲染PNG背景**: matplotlib KDE热力图 → base64 PNG, 零polygon trace
  - **Dominance-peak标签**: 每个领地标签放在该领地"最独占"的位置(自身密度-他人密度最大处), 解决标签堆叠中心的问题
  - **层级缩放 (Overview/Detail)**: 两个按钮切换field标签(5个)和subfield标签(8个), 类Google Maps层级
  - **稳定背景缓存**: 首次1.8s, 后续filter/时间滑块复用缓存0.14s
  - Pastel色系 (soft blue/salmon/green/purple/grey), 低饱和度
  - "海洋"效果: 低密度透明 + gaussian_filter平滑边缘
  - 6个total traces (1 Scattergl + 5 legend), 点3-7px opacity 0.35
  - 时间演化: 共享build_territory_scatter, 背景缓存

### Added
- **Paper Landscape 时间演化 ("活泡泡")** — 年份滑块 + Play/Pause 自动播放
  - 单年份滑块: 显示"up to {year}"的累积论文
  - 每个cluster画一个半透明泡泡 (centroid位置, 大小=√论文数×12)
  - 泡泡上显示cluster标签+论文数, 在散点下层
  - Play按钮: 800ms间隔从最早年份自动播放到最新, 到达终点自动停止
  - 动态标题: "Research Landscape — up to {year}"
- **多作者叠加对比** — 选多个学者叠加在同一UMAP空间
  - "Add to Compare"按钮: 将当前选中作者加入对比列表 (最多5人)
  - 所有论文在同一UMAP空间embed+投射+KMeans聚类
  - 双重编码: 填充色=研究领域/cluster, marker形状+边框色=作者身份
    - 形状: circle, diamond, square, triangle-up, cross
    - 边框: gold, #FF6692, #19D3F3, #B6E880, #FF97FF
  - 合著论文: 大白星标记 (star, size=18, white, gold border)
  - Legend按作者分组, 合著论文单独条目
  - Split View模式: 每个作者一个子图, 坐标轴共享, 合著论文在每个panel中标星

## [0.3.0] - 2026-03-28

### Added
- **Research Territory标签改进** — OpenAlex topic/subfield聚合替代TF-IDF
  - 粗层: subfield多数投票 (如 "Strategy and Management")
  - 细层: topic多数投票 (如 "Digital Platforms and Economics")
  - TF-IDF保留作为fallback (论文无topic时)
- **3D Scholar Temporal Trajectory** — 分叉树可视化
  - X/Y = UMAP研究空间, Z = publication year
  - 每篇论文dot按所属cluster着色, 大小按citation log缩放(4-22)
  - **分叉树结构 (phylogenetic branching)**:
    - Spine: 每cluster内按年份串连最高引论文, 线端点=真实论文
    - Fork: 新cluster首次出现时从最近论文位置自然分叉
    - Spoke: 同年多篇论文时, 非spine论文用辐条连到spine节点
  - 背景: co-author论文作为极淡cluster色小点 (opacity=0.12)
  - 底面投影阴影
  - Legend显示各cluster名称+颜色
  - 预设视角按钮: 3D View / Top Down / Side View
  - turntable旋转保持年份轴竖直
- **Inner Circle检测** — 核心合作圈子识别
  - 复合评分: 频率(log) + 时效性(指数衰减) + Jaccard排他性
  - 最大断崖法自动划分inner circle边界, 最少共著>=2篇
  - Ego network可视化区分: inner circle金色边框+名字标签, periphery普通蓝色
  - Inner circle之间白色加粗连线, focal↔inner circle金色加粗连线
- **Inner Circle研究画像** — 圈子集体profile面板
  - 成员列表: 名字、机构、共著数、评分
  - Circle density (紧密度): actual_edges/possible_edges
  - 集体研究方向: inner circle全体论文的top 5 OpenAlex topics

## [0.2.0] - 2026-03-28

### Added
- **Author Landscape** — 960万学者数据, citation-weighted embedding, KMeans聚类
  - 搜索学者名字 → 高亮定位(gold star)
  - 共著网络连线(白色半透明)
  - 点击学者 → 论文列表、合作者、机构
  - Research Territory视图 + 层级嵌套椭圆(AgglomerativeClustering + Ward)
- **Author表** — authors(960万) + paper_authors(2490万条链接)
- `embed.py`: `generate_author_embeddings()` citation加权平均
- `embed.py`: `generate_embeddings_from_texts()` 内存直接embedding
- `cluster.py`: `cluster_from_embeddings()` 内存KMeans+UMAP
- `cluster.py`: `cluster_authors()` 学者聚类
- `app.py`: Paper Landscape / Author Landscape 模式Tab切换
- `app.py`: "Build Author Landscape" 按钮

## [0.1.0] - 2026-03-27

### Added
- **智能搜索** — Claude API tool_use + SQLite FTS5全文检索
  - 用户输入自然语言 → Claude自动生成搜索策略 → FTS5查询 → 结果可视化
  - `search_papers` + `get_db_stats` 两个tool
  - Agentic loop (最多15轮tool调用)
- **SQLite数据库** — 2360万篇IS相关论文, 51.3GB
  - 从OpenAlex snapshot (583GB) 流式构建
  - FTS5全文检索索引 (title + abstract)
  - Journal tier分级: Tier1-FT/UTD, Tier2-Basket8, Tier3-Strong
- **Dash Web应用** — DARKLY主题, 端口8050
  - API Key输入框 (localStorage持久化)
  - 散点图: 缩放/平移/hover/点击
  - 右侧面板: Paper Details / AI Cluster Analysis / AI Landscape
  - 过滤器: 年份/citation/cluster
  - 导出: CSV/BibTeX/PNG(300dpi via kaleido)
- `download_openalex.py` — snapshot下载(boto3) + 流式提取
- `build_db.py` — 流式SQLite构建(低内存) + FTS5 + author表
- `analyze.py` — tool_use智能搜索 + cluster/landscape分析
- `run_pipeline.bat` — 一键启动, 自动杀旧进程+清缓存

### Technical Decisions
- 本地SQLite替代OpenAlex API (API限流不可用)
- Claude tool_use替代MCP (单应用场景,MCP过度工程)
- Journal ranking: FT50/UTD24 + AIS Basket of 8 + ABDC (学术共识)
