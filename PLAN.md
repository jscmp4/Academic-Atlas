# Research Landscape Mapping System — Master Plan

## 项目愿景

构建一个交互式学术研究景观地图系统，用户输入自然语言研究想法 → LLM智能搜索本地论文数据库 → Embedding + 聚类 → 交互式可视化。支持论文级别和**学者级别**的景观映射。

---

## 当前已完成

### 基础设施 ✅
- OpenAlex snapshot已下载到 `N:/openalex-snapshot/`（583GB，2236个.gz文件）
- SQLite数据库正在构建中 `data/openalex.db`（预计72万+篇IS相关论文）
- 数据库schema: openalex_id, title, abstract, citation_count, year, journal, doi, authors, subfield, field, topic, journal_tier
- Journal tier分级已按学术共识定义：
  - **Tier1-FT/UTD**: MISQ, ISR, Management Science
  - **Tier2-Basket8**: JMIS, JAIS, EJIS, ISJ, JIT, JSIS
  - **Tier3-Strong**: I&M, DSS, CACM, CHB等30+期刊
  - **Other**: 其余IS相关期刊

### Pipeline ✅
- `fetch.py` — OpenAlex API采集（有rate limit问题，已被snapshot方案替代）
- `embed.py` — sentence-transformers embedding（MiniLM/mpnet/SPECTER2可切换）
- `cluster.py` — BERTopic (UMAP+HDBSCAN) / KMeans fallback
- `download_openalex.py` — snapshot下载+流式提取
- `build_db.py` — 流式构建SQLite（低内存）

### Web应用 ✅（基础版）
- `app.py` — Plotly Dash，深色主题(DARKLY)，端口8050
- 散点图：缩放/平移/hover/点击
- 右侧面板：Paper Details / AI Cluster Analysis / AI Landscape
- 过滤器：年份/citation/cluster
- 导出：CSV/BibTeX/PNG
- AI Interpret：Claude API生成搜索参数
- 自动PID管理（data/app.pid）

### 已验证可工作
- 25,426篇论文（Tier1+2），405个cluster，embedding+clustering全流程跑通
- Dash可视化正常显示

---

## 待实现

### Phase 1: 智能搜索（Claude Tool Use + SQLite FTS5）

**目标**：用户输入自然语言 → Claude自动搜索本地数据库 → 返回相关论文

**技术方案**：
- SQLite FTS5全文检索索引（内置，零依赖）
- Anthropic API tool_use（Claude自动决定搜索参数）
- 用户提供自己的Anthropic API Key

**需要修改的文件**：
1. `build_db.py` — 添加FTS5虚拟表
   ```sql
   CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract, content=papers, content_rowid=rowid);
   INSERT INTO papers_fts(papers_fts) VALUES('rebuild');
   ```

2. `analyze.py` — 添加tool_use搜索函数
   - 定义 `search_papers` tool schema（keywords, year_min, year_max, citation_min, journal_tiers, subfields, limit）
   - 实现 `execute_search()` 函数（FTS5查询 + 结构化过滤）
   - 实现 `intelligent_search()` 函数（Claude agentic loop：发消息→tool_use→执行→返回）

3. `app.py` — 重写AI Research Assistant面板
   - API Key输入框（存到session/localStorage）
   - 研究想法输入框
   - 点击搜索 → 调用intelligent_search → 显示搜索结果
   - "Visualize these papers" 按钮 → embed + cluster选中论文 → 更新散点图
   - 整个流程同步执行，dcc.Loading显示spinner

**Tool定义**：
```python
{
    "name": "search_papers",
    "description": "Search the IS academic papers database (700K+ papers). ...",
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}},
            "year_min": {"type": "integer"},
            "year_max": {"type": "integer"},
            "citation_min": {"type": "integer"},
            "journal_tiers": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"}
        },
        "required": ["keywords"]
    }
}
```

### Phase 2: 学者景观映射（Author Landscape）

**目标**：输入学者名字 → 看到他在研究版图上的位置 → 他的小圈子 → 和谁有联系

**这是一个学术贡献，现有工具都没做到这个**（调研过18+工具）。

**技术方案**：

1. **Author Embedding**：
   - 从SQLite取某学者的所有论文
   - 取这些论文的embedding，计算平均值 → 得到该学者的"研究身份向量"
   - 或者用加权平均（citation-weighted）

2. **Author数据库**：
   - 从OpenAlex snapshot的authorships字段提取作者信息
   - `build_db.py` 添加 `authors` 表和 `paper_authors` 关联表
   ```sql
   CREATE TABLE authors (
       author_id TEXT PRIMARY KEY,
       name TEXT,
       institution TEXT,
       paper_count INTEGER,
       total_citations INTEGER
   );
   CREATE TABLE paper_authors (
       paper_id TEXT,
       author_id TEXT,
       PRIMARY KEY (paper_id, author_id)
   );
   ```

3. **Author Landscape可视化**：
   - 所有活跃作者（比如>=3篇论文）→ 计算author embedding → UMAP 2D → 聚类
   - Dash新Tab: "Author Landscape"
   - 每个点=一个学者，颜色=研究cluster，大小=总citation
   - 搜索框：输入名字 → 高亮定位该学者 → 显示他的论文、合作者、所属cluster
   - 点击一个cluster → 显示这个小圈子的所有人

4. **Co-authorship网络**：
   - 从paper_authors表构建共著关系
   - 点击一个学者 → 显示他的直接合作者（用线连接）
   - 或者在散点图上用线段连接共著者

**需要修改的文件**：
- `build_db.py` — 添加authors表，提取作者数据
- `embed.py` — 添加author embedding功能
- `cluster.py` — 添加author clustering
- `app.py` — 添加Author Landscape tab

### Phase 3: MVP打包

**目标**：让任何用户都能用

1. **用户只需提供**：
   - Anthropic API Key
   - （可选）自己的OpenAlex数据库路径

2. **首次使用流程**：
   - 启动app.py → 网页端输入API Key
   - 输入研究想法 → Claude搜索 → 结果可视化
   - 点击作者名 → 看到作者景观

3. **技术细节**：
   - API Key存在浏览器localStorage（不传到服务端存储）
   - 每次API调用时从前端传Key
   - 预构建的SQLite数据库随项目分发（或提供build脚本）

---

## 文件结构（最终）

```
research-landscape/
├── PLAN.md                  # 本文件
├── app.py                   # Dash Web应用（主入口）
├── analyze.py               # LLM分析（tool_use搜索 + cluster分析 + landscape分析）
├── build_db.py              # 构建legacy SQLite（IS论文子集）
├── extract_to_parquet.py    # Lakehouse Step 1: snapshot → Parquet（全量2.97亿）
├── build_derived.py         # Lakehouse Step 2: Parquet → DuckDB/SQLite FTS5
├── download_openalex.py     # snapshot下载+流式提取
├── embed.py                 # Paper/Author embedding
├── cluster.py               # Paper/Author clustering
├── utils.py                 # 工具函数
├── config.yaml              # 搜索配置
├── config_default.yaml      # 默认模板
├── .env                     # API keys
├── requirements.txt
├── run_pipeline.bat         # Windows快捷启动
├── data/
│   ├── openalex.db          # Legacy SQLite（2300万IS论文 + 作者表）
│   ├── papers.csv           # 当前工作子集
│   ├── papers_clustered.csv # 聚类结果
│   ├── embeddings.npy       # Paper embeddings
│   ├── app.pid              # 进程PID文件
│   └── cache/               # API查询缓存
├── output/                  # 导出文件
│
├── N:/openalex-snapshot/    # 原始snapshot（583GB，2236个.gz）
└── N:/academic-data/        # Lakehouse Lite
    ├── parquet/             # 唯一真实数据源
    │   ├── works_YYYY.parquet       # 年份分区，全量2.97亿篇
    │   ├── authorships_YYYY.parquet # 论文-作者关联
    │   └── authors.parquet          # 去重后的学者表
    └── derived/             # 可随时重建的派生层
        ├── analytics.duckdb         # DuckDB分析库
        └── search.db                # SQLite FTS5搜索
```

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 数据源 | OpenAlex snapshot (本地) | API限流太严重，本地无限制 |
| 存储架构 | Parquet数据源 + 派生查询层 | 列式压缩80GB vs 700GB，schema evolution，分析快10-100x |
| 分析查询 | DuckDB | 原生读Parquet，列式扫描，亿级聚合秒级响应 |
| 全文搜索 | SQLite FTS5 | BM25排序，零依赖，列名兼容旧schema |
| LLM集成 | Anthropic API tool_use | 比MCP简单，适合单应用场景 |
| Embedding | all-MiniLM-L6-v2 (默认) | CPU友好，384维，速度快 |
| 聚类 | BERTopic (UMAP+HDBSCAN) | 自动topic标签，业界标准 |
| 前端 | Plotly Dash + Bootstrap | Python全栈，交互性好 |
| 期刊分级 | FT50/UTD24 + Basket8 + ABDC | 学术界共识 |

---

## 协作模式

本项目采用**双session协作**：

```
用户（决策者）
  ↕  讨论需求和方向
架构session（规划层）
  │  - 调研可行性（搜网、验证工具、评估方案）
  │  - 设计技术方案（写到PLAN.md）
  │  - 审查实现session的产出
  │  - 维护memory（跨session持久记忆）
  ↓  更新PLAN.md
实现session（执行层）
  │  - 读PLAN.md了解任务
  │  - 写代码、debug、测试
  │  - 遇到架构问题 → 用户带回架构session讨论
  ↓
代码产出 → 用户验收 → 架构session更新PLAN.md状态
```

**PLAN.md是两个session之间的接口协议。** 架构session写，实现session读。

**加新feature的流程：**
1. 用户跟架构session讨论想法
2. 架构session调研 → 出方案 → 用户确认
3. 架构session更新PLAN.md（具体到改哪些文件、用什么技术、注意什么）
4. 用户开新session："读PLAN.md，实现Phase X"
5. 实现完毕 → 用户验收 → 架构session标记完成、规划下一步

---

## 当前状态

→ 已完成的功能见 [CHANGELOG.md](CHANGELOG.md)
→ 待做功能和优先级见 [BACKLOG.md](BACKLOG.md)

---

## 给实现AI的指引

如果你是一个新的AI session来实现这个计划：

### 首先做这些
1. **读这些文件了解全貌**：`PLAN.md`（本文件）, `app.py`, `analyze.py`, `build_db.py`, `utils.py`
2. **检查数据库状态**：
   ```bash
   python -c "import sqlite3; c=sqlite3.connect('data/openalex.db'); print('Papers:', c.execute('SELECT COUNT(*) FROM papers').fetchone()[0])"
   ```
3. **检查FTS5是否已建**：
   ```bash
   python -c "import sqlite3; c=sqlite3.connect('data/openalex.db'); print(c.execute(\"SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH 'machine learning'\").fetchone()[0])"
   ```

### 关键约束
- **Dash主题**：DARKLY（深色），端口8050
- **PID管理**：app.py启动时写data/app.pid，下次启动时杀旧进程
- **run_pipeline.bat**：每次启动前自动清__pycache__和杀旧PID
- **内存**：全量数据2400万行，绝对不能pandas一次读取，必须用SQLite查询
- **API Key**：用户自己的Anthropic key，从前端传入，不hardcode
- **不要用后台线程+轮询做进度**：Dash callback冲突太多，用dcc.Loading同步执行即可
- **suppress_callback_exceptions=True** 已开启（动态生成的组件需要这个）

### 已有代码的关键更新（实现session已做）
- `analyze.py`：已添加 `intelligent_search()` + tool_use agentic loop + `execute_search()` FTS5查询
- `build_db.py`：已添加 `--build-fts` 命令构建FTS5索引
- 这些代码已经可以工作，下一步是集成到app.py的前端
