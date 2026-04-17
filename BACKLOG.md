# Backlog

## Currently Building

### Author 对比模式
- 两个（或多个）学者的研究领地并排/叠加对比
- 已有基础：多作者叠加 Compare 功能（v0.4.0），需要优化交互

## Up Next

- **研究空白检测** — 点击地图空白区域 → LLM 分析潜在交叉研究方向
- **Semantic search** — 除了 FTS5 关键词搜索，加 embedding 向量相似度搜索
- **Citation 网络** — 论文间的引用关系图 (referenced_works 字段)

## Ideas (未排期)

- **推荐系统** — "和你研究相似的学者", "你可能感兴趣的论文" (基于 embedding 相似度)
- **用户标记论文** — data/user_papers.csv, 地图上★标记
- **部署方案** — Docker/云端部署供他人使用
- **CT 扫描优化** — 时间切片 + deck.gl filterSoftRange 渐变动画改进

## Completed

- [x] **Lakehouse Lite** (2026-04-01) — Parquet 455M + DuckDB 453GB + SQLite FTS5 251M + authors 1.09亿
- [x] **World Map** (2026-04-04) — 全学科 50K 高引论文, sqrt-proportional 采样, 157 clusters
- [x] **CT 扫描时间滑块** (2026-04-04) — deck.gl DataFilterExtension, GPU 过滤, Play/Cumulative 模式
- [x] **标签优化** (2026-04-04) — OpenAlex topic 多数投票替代 TF-IDF, 去重, 405 唯一标签
- [x] **Author 消歧确认** (2026-04-04) — Search → 候选列表 → Confirm & Build 按钮
- [x] **去掉边界线** (2026-04-04) — cluster_boundary_polygons=False
- [x] **v0.1.0** (2026-03-27) — 智能搜索 + SQLite + FTS5 + Dash Web 应用
- [x] **v0.2.0** (2026-03-28) — Author Landscape + Research Territory + 层级椭圆
- [x] **v0.3.0** (2026-03-28) — 3D 分叉树轨迹 + Inner Circle 检测 + 圈子画像
- [x] **v0.4.0** (2026-03-28) — 时间演化活泡泡 + 多作者叠加对比
- [x] **Paper Landscape 领地化** (2026-03-28) — 405 clusters → 16 KDE 领地 + Scattergl
- [x] **Paper Landscape datamapplot** (2026-03-28) — Plotly → datamapplot DeckGL 语义缩放
