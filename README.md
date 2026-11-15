# 特征平台 (Feature Platform)

一个可以真正跑起来的风控特征平台：**特征定义一次，离线批量与在线实时两套逻辑自动生成**，并且从数据生成、训练样本、在线服务、一致性比对到漂移监控全部端到端可验证。

> 解决的核心问题：离线训练用的特征和线上实时算的特征对不上，模型上线效果就掉。
> 本平台用「声明式定义 + 双编译器 + 点时间语义 + 抽样比对 + 泄漏审计」把这四件事钉死。

```
                       ┌───────────────────────────────┐
                       │  feature_platform/features/    │
                       │        specs.yaml             │  ← 唯一定义源
                       └───────────┬───────────────────┘
                                   │ 同一份 spec
                 ┌─────────────────┴──────────────────┐
                 ▼                                    ▼
     ┌───────────────────────┐            ┌────────────────────────┐
     │ offline/compiler.py   │            │ online/engine.py       │
     │ → DuckDB SQL          │            │ → NumPy 逐用户求值      │
     │   训练样本 / 批量回填  │            │   在线单用户查询         │
     └───────────┬───────────┘            └───────────┬────────────┘
                 │                                    │
                 └──────────────┬─────────────────────┘
                                ▼
                  audit/parity.py  ·  monitor/consistency.py
                  （同一用户同一时刻两条链路逐特征比对）
```

---

## 快速开始

```bash
# 环境：Python 3.11+，依赖见 requirements.txt
pip install -r requirements.txt

# 一键构建全部内容（造数 → 冻结编码表 → 训练样本 → 一致性 → 泄漏审计 → 漂移 → 模型评估）
python scripts/build_all.py            # 约 3 分钟
python scripts/build_all.py --quick    # 1 分钟，采样更小

# 启动管理界面与 API
uvicorn feature_platform.api.app:app --port 8000
# 打开 http://localhost:8000
```

单独运行某一步：

```bash
python -m feature_platform.data.generate              # 造两年数据（含时间陷阱）
python -m feature_platform.online.encodings --observe # 冻结类别编码表
python -m feature_platform.offline.samples            # 生成训练样本 parquet
python -m feature_platform.audit.parity --n 2000 --mode both
python -m feature_platform.audit.leak --n-prefix 60 --n-boundary 25 --n-naive 2000
python -m feature_platform.audit.model_eval
python -m feature_platform.monitor.consistency --n 500 --mode both
python -m feature_platform.monitor.drift --days 150
python -m pytest -q                                    # 93 个测试
```

---

## 一、声明式特征定义（需求 1）

`feature_platform/features/specs.yaml` 是唯一定义源。每个特征只写一次，由两个编译器各自消费：

```yaml
- name: loan_amt_sum_30d
  kind: window              # window | asof_attr | static_attr | derived | constant
  source: loans
  window: 30d
  agg: sum
  column: principal
  type: float
  fill: 0.0
  tags: [borrow, window]

- name: credit_score        # SCD2 快照，取 as_of 之前最后一版
  kind: asof_attr
  source: profile
  attr: credit_score

- name: loan_to_app_ratio_30d   # 派生，可引用其他特征
  kind: derived
  expr: "case when app_cnt_30d > 0 then loan_cnt_30d / app_cnt_30d else 0.0 end"
  fill: 0.0
```

刻画一个数据源时，必须声明它的**时间语义**——这是整个平台的地基：

```yaml
sources:
  loans:
    kind: event
    table: loans
    event_time: loan_time          # 业务发生时间
    knowledge_time: ingested_at    # 数仓何时知道的（迟到数据、补数）
    valid_until: cancelled_time    # 何时失效（撤销/退款/作废）
    dedupe_key: loan_id            # 仅追加日志里的重复行
    dedupe_order: ingested_at      # 取 as_of 当时最新的那一版
    forbidden: [final_dpd]         # 结果列，永远不能当特征
```

编译器产物（离线 SQL）与在线执行计划都从这份声明推导，没有任何一处逻辑是手写两遍的。

### 双编译器

| | 离线 `offline/compiler.py` | 在线 `online/engine.py` |
|---|---|---|
| 输出 | 一条 DuckDB SQL，每个窗口特征一个 `LEFT JOIN LATERAL` | 逐用户 NumPy 向量求值 |
| 用途 | 训练样本、批量回填、离线对账 | 单用户毫秒级查询 |
| 数据 | 列式全表 | 启动时按 entity 建倒排索引，`searchsorted` 定位时间窗 |
| 语义 | SQL 三值逻辑 | 同一套三值逻辑（`expr.py`） |

`expr.py` 是一个小型表达式引擎：同一表达式既 `to_sql()` 也 `evaluate()`，NULL 传播、Kleene 逻辑、除零、字符串转义、`nullif` 等都按标准 SQL 语义实现，并由一组随机对照测试保证两个目标一致。

---

## 二、点时间语义：训练样本绝不使用申请之后的数据（需求 2）

每条源数据要进入 `as_of` 时刻的特征，必须同时满足：

```
event_time      <  as_of                业务上已经发生（严格小于）
knowledge_time  <= as_of                数仓当时已经知道（挡住迟到数据）
valid_until     >  as_of 或 NULL        当时还没有被撤销（作废/退款）
deferred 列      known_at <= as_of      当时还不知道的结论（如审批结果）
```

`as_of` 就是训练样本的 `apply_time`。这四条规则在两个引擎里各自独立实现——正因为独立实现，一致性比对才有意义。

### 数据里埋的时间陷阱

造数脚本 `feature_platform/data/generate.py` 刻意埋了 8 类陷阱：

| # | 陷阱 | 会打穿哪种错误实现 |
|---|---|---|
| T1 | `apply_time_local` 是 +08:00 字符串 | 直接 CAST 当 UTC 用，事件时间整体偏移 8 小时 |
| T2 | `apply_status`/`reject_reason` 在 `decision_time` 才产生 | 用申请结果做历史特征 |
| T3 | `loans` 是追加日志，8% 的 loan 会被重述 | 不去重，金额/笔数翻倍 |
| T4 | `loan_status='CANCELLED'` 是未来才确定的 | 回看时把已作废的贷款算进去 |
| T5 | 还款可被撤销（`reversal_time`） | 把已冲正的还款算进"已还金额" |
| T6 | `user_profile_history` 是 SCD2 快照 | 用最新画像而不是 as_of 快照 |
| T7 | 合作行/线下渠道 1–45 天才回传 | 按事件日期分区而不是按摄入时间，回补数据提前出现 |
| T8 | 标签有 7 天结算延迟 + 90 天观察窗 | 用还没成熟的标签训练 |

两年数据规模：24,000 用户 / 51,263 申请 / 32,513 贷款 / 190,860 还款 / 30,023 标签。

---

## 三、滑动窗口特征（需求 3）

近 7 天 / 30 天 / 90 天 / 180 天 / 365 天的借款笔数与金额、申请笔数与金额、还款计划、逾期笔数等，共 33 个特征。窗口是**左闭右开**的 `[as_of - window, as_of)`，即严格不含 `as_of` 当下。

```
loan_cnt_7d     loan_amt_sum_7d     loan_cnt_30d    loan_amt_sum_30d
loan_amt_avg_180d  loan_cnt_365d   loan_interest_max_90d   loan_last_principal
app_cnt_7d / 30d / 180d    app_amt_sum_30d    app_amt_avg_90d
app_distinct_product_90d   app_approved_cnt_90d   rejected_cnt_180d
overdue_instalments_cnt    scheduled_due_amt_30d  paid_amt_sum_30d / 90d
```

---

## 四、在线查询延迟（需求 4）

单用户全量 33 个特征的进程内计算：

| 指标 | 值 |
|---|---|
| 引擎内 p50 | **~190 µs** |
| 引擎内 p95 | **~600 µs** |
| HTTP 端到端 p50 / p95 | **0.9 ms / 1.3 ms** |
| 缓存命中时 HTTP | **0.5 ms** |

实现方式：启动时把源表按 `entity_id` 建成列式索引（每实体一个按事件时间排序的 `EntityFrame`），查询用 `np.searchsorted` 定位时间窗，再用向量化掩码套用四条点时间规则。索引常驻内存，19,403 个活跃用户约 3 秒建完。

```bash
curl -X POST localhost:8000/v1/score -H 'Content-Type: application/json' -d '{
  "view": "user_risk_v1", "entity_id": "U0000001", "as_of": "2025-06-01T12:00:00"
}'
```

```json
{
  "entity_id": "U0000001",
  "as_of": "2025-06-01 12:00:00",
  "features": {"loan_cnt_30d": 0, "credit_score": 731, "risk_level_code": 1, ...},
  "encoded":  {"risk_level_code": 1, "city_tier_code": 2, ...},
  "category_raw": {"risk_level_code": "medium", ...},
  "unknown_category": {"risk_level_code": false, ...},
  "latency_us": 214.7,
  "cache_hit": false
}
```

---

## 五、离线 / 在线抽样比对（需求 5）

`audit/parity.py` 与 `monitor/consistency.py` 对同一 `(user, as_of)` 分别调用两条链路，逐特征比对：

| 场景 | 比对量 | 不一致 cell 数 | 不一致率 |
|---|---:|---:|---:|
| 训练样本时刻（`as_of = apply_time`），2000 行 × 33 特征 | 66,000 | **0** | **0.000000%** |
| 在线监控随机时刻（含无申请的时刻），800 次 × 33 特征 | 26,400 | **0** | **0.000000%** |
| 抖动 ±2 秒（模拟请求时间轻微偏移） | 66,000 | 274 | 3.03% |

**抖动场景的 274 个不一致全部可解释**：每一个不一致的 key，其 ±2 秒窗口内都恰好有一条源事件，即该事件跨过了 `as_of` 边界。这不是流水线缺陷，是"同一时刻"定义的边界效应；报告里会单独统计 `skew_mismatches_unexplained`，当前为 **0**。

容差：浮点 `1e-9` 相对误差，整数/字符串/布尔精确相等，`NULL == NULL`。

界面上可直接触发比对（`/ui/consistency`），历史结果存 `monitor_consistency` 表。

---

## 六、缺失率与分布漂移按天监控（需求 6）

`monitor/drift.py` 每天对每个特征计算：

| 指标 | 方法 | 阈值 |
|---|---|---|
| `missing_rate` | 当日 vs 过去 90 天，Wilson 区间 + 最小差值 2% | 需同时显著且幅度够大 |
| `psi` | 以参考窗分位数为桶 | ≥ 0.25 告警 |
| `ks_stat` | 两样本 KS + 渐近 p 值 | p < 0.01 告警 |
| `mean_shift_sd` | 均值位移 / 参考标准差 | \|z\| ≥ 0.25 |
| `unknown_category_share` | 落到 `__UNKNOWN__` 的占比（冻结编码） | ≥ 2% 告警 |

缺失率高的特征（如 `loan_interest_max_90d`、`loan_last_principal`）自动跳过 PSI/KS，因为这类分布指标在大量 NULL 下没有意义——由缺失率指标独立覆盖。

数据里刻意做了 2025 年下半年的风险结构漂移，监控能抓到：141 个评估日中，`days_since_first_app`、`credit_score`、`account_age_days`、`loan_interest_max_90d` 等特征的 PSI/KS 持续越界，与造数时的风险抬升设计一致。

---

## 七、类别编码固化与兜底（需求 7）

`online/encodings.py` 把训练窗口观察到的类别字典**冻结**成 `data/online_index/encodings.json`：

```json
"user_risk_v1.city_tier_code": {
  "dictionary": ["tier1", "tier2", "tier3", "tier4", "__UNKNOWN__"],
  "mapping": {"tier1": 0, "tier2": 1, "tier3": 2, "tier4": 3, "__UNKNOWN__": 4},
  "default": "__UNKNOWN__",
  "default_code": 4,
  "dictionary_hash_sha256_16": "453c9e45e41bbc41"
}
```

线上**永不重建**字典。遇到新类别：

1. 映射到显式的 `__UNKNOWN__` 桶（不会 KeyError、不会隐式扩维度）；
2. 响应里 `unknown_category.<feature> = true`，进程内计数暴露到 `/v1/health`；
3. 漂移监控里的 `unknown_category_share` 持续上升即提示编码表该随模型一起发版了。

离线/在线用同一份冻结表，因此编码结果天然一致。

---

## 八、特征缓存与source级失效（需求 8）

`online/cache.py` 的缓存粒度是**单个特征值**，不是整个向量：

- key: `(spec指纹, view, feature, entity, as_of桶)`
- 每条缓存记录它读过哪些源（由 spec 的依赖图推导，派生特征继承输入的源）
- source 版本号 +1 时，只淘汰依赖该源的条目

实测：bump `profile` 只淘汰 33 个特征中的 3 个（`credit_score` / `risk_level_code` / `is_blacklisted`），不依赖 profile 的申请历史、借款历史全部命中。

其他机制：TTL 按视图配置（默认 300s）对 `as_of` 分桶；same-key 请求单飞合并（single-flight）；LRU 兜底。

界面 `/ui/views` 可以点按钮演示失效范围，`/v1/monitor/latency` 返回命中率与淘汰原因计数。

---

## 九、泄漏审计：证明训练样本没有用到未来数据

`audit/leak.py` 用四个互相独立的证明。这是平台最重要的产出。

### 证明 1：前缀稳定性（数据访问级证明）

把整个数据仓库**物理截断**到 `as_of` 时刻（每张源表都加
`event_time <= as_of AND knowledge_time <= as_of`），用同一份 compiler 重算特征；
结果必须与全量仓库在同一天算出的完全一致。

> 如果一个特征偷看了任何未来行，截断后必然会变。
> 该证明不读特征代码，只减少数据，因此不依赖实现是否正确。

**结果：1,980 个 cell，0 处不一致。**

### 证明 2：边界变异（off-by-one 探测）

把 `as_of` 之后、或 `as_of` 时尚未摄入的每一行源数据，其数值列改成一个夸张的值。特征不得有任何变化。

**结果：825 个 cell，0 处不一致。**

### 证明 3：朴素实现对照（正向对照）

用同一份 spec，换成"手写 SQL 常见做法"的编译模式，量化差异：

| 朴素变体 | 变化的 cell 占比 | AUC |
|---|---:|---:|
| `inclusive_boundary`（窗口用 `<=`） | 2.2% | 0.6622 |
| `knowledge_lag_ignored`（忽略摄入延迟） | 2.7% | 0.6575 |
| `no_dedupe`（重复行不去重） | 0.3% | 0.6425 |
| `no_validity`（作废/冲正仍计入） | 0.3% | 0.6554 |
| `latest_profile`（用最新画像） | 4.0% | 0.5737 |
| `local_time_as_utc`（本地时间当 UTC） | 2.2% | 0.6524 |
| `deferred_unguarded`（审批结果提前用） | 0.5% | 0.6353 |
| **全部朴素措施叠加** | **28.3%** | 0.6555 |

### 证明 4：反向对照（每条规则必须"可被检测到"）

逐条去掉一个点时间规则，**该规则必须导致至少一个特征变化**。若某条规则去掉后毫无变化，说明审计本身对该规则是瞎的。

| 去掉的规则 | 变化的 cell | 受影响特征 |
|---|---:|---|
| `no_dedupe` | 5 | loan_amt_avg_180d, loan_cnt_365d, loan_last_principal |
| `no_validity` | 3 | loan_amt_avg_180d, loan_cnt_365d |
| `latest_state` | 118 | credit_score, risk_level_code |
| `knowledge_lag` | 97 | amt_per_app_30d, app_amt_avg_90d, app_amt_sum_30d |
| `inclusive_boundary` | 87 | amt_per_app_30d, app_amt_avg_90d, app_amt_sum_30d |
| `local_time_shift` | 87 | amt_per_app_30d, app_amt_avg_90d, app_amt_sum_30d |
| `no_deferred` | 37 | has_overdue_flag, overdue_instalments_cnt, paid_amt_sum_30d |

**7/7 全部可检测 → `all_detected: true`。**

### 结论

`reports/leak_audit.md` 的 verdict 为 **PASS**：证明 1、2 无违规，证明 4 全部规则可检测。

---

## 十、模型效果：泄漏会把离线 AUC 抬到什么程度

`audit/model_eval.py` 在时间切分（train < 2024-11 < valid < 2025-04 ≤ test）上训练同一个 GBDT：

| 特征集 | Test AUC |
|---|---:|
| **点时间特征（平台产物）** | **0.6906** |
| + 未来 90 天窗口聚合（典型的"忘了加 as_of 条件"） | **0.9953** |
| 全部朴素实现的特征 | 0.6627 |

- 未来窗口泄漏把 AUC 从 **0.69 抬到 0.995**（+0.30）——这就是"离线看起来神准、上线就掉"的量化来源。
- 朴素实现本身并不保证抬高 AUC（本例 -0.028），因为它既泄漏又引入噪声；它的危害在于**特征值本身是错的**，模型学到的是不可复现的关系。这正是为什么需要证明 1–4，而不是只看 AUC。

---

## 十一、目录结构

```
feature_platform/
├── config.py                 全局配置、时间常量
├── expr.py                   双目标表达式引擎（SQL + NumPy）
├── spec.py                   特征/视图/数据源模型、校验、编码表
├── features/specs.yaml       ★ 唯一定义源（33 个特征 / 5 类数据源）
├── data/
│   ├── schema.sql            源表结构
│   └── generate.py           两年造数 + 8 类时间陷阱
├── offline/
│   ├── compiler.py           spec → 点时间 DuckDB SQL（含朴素变体）
│   └── samples.py            训练样本 + 时间切分
├── online/
│   ├── engine.py             在线求值 + 实体倒排索引
│   ├── cache.py              逐特征缓存 + 依赖感知失效
│   └── encodings.py          类别字典冻结 CLI
├── audit/
│   ├── parity.py             离线/在线逐特征比对
│   ├── leak.py               四个泄漏证明
│   └── model_eval.py         干净/泄漏/朴素三种特征集的 AUC
├── monitor/
│   ├── consistency.py        随机时刻抽样比对 + 持久化
│   └── drift.py              缺失率/PSI/KS/均值位移
├── api/
│   ├── app.py                FastAPI 路由 + 6 个管理页面
│   └── state.py              单例服务状态、延迟统计
└── web/                      模板与样式
tests/                        93 个测试
scripts/build_all.py          一键构建
reports/                      parity / leak_audit / consistency / drift / benchmark
```

### 管理界面

| 页面 | 路径 | 内容 |
|---|---|---|
| Overview | `/` | 特征数、在线延迟、一致率、缓存命中率、泄漏审计结论、在线试算 |
| 特征定义 | `/ui/features` | 数据源时间语义、33 个特征定义、编译出的 SQL |
| 特征视图 | `/ui/views` | 依赖图、编译语句、缓存失效演示 |
| 训练样本 | `/ui/training` | 切分统计、四个证明、朴素对照、样本预览 |
| 在线监控 | `/ui/monitoring` | 延迟分位数、缓存、冻结编码表、漂移越界 |
| 一致性检查 | `/ui/consistency` | 实时触发比对、逐特征不一致、历史运行 |

---

## 十二、主要 API

```
POST /v1/score                     单用户特征向量
POST /v1/score/batch               批量
GET  /v1/views                     视图目录
GET  /v1/views/{view}/features     特征元数据 + 编译 SQL
GET  /v1/views/{view}/encodings    冻结编码表
GET  /v1/health                    延迟、缓存、未知类别计数
GET  /v1/monitor/latency           延迟分位数
GET  /v1/monitor/consistency       一致性历史
POST /v1/monitor/consistency/run   触发一次比对
GET  /v1/monitor/drift             漂移时间序列
GET  /v1/audit/leak                最近一次泄漏审计
GET  /v1/audit/parity              最近一次一致性报告
POST /v1/admin/reload              重载 spec 与索引
POST /v1/admin/invalidate?source=  令某个源的缓存失效
```

---

## 十三、设计取舍

**为什么两个编译器是独立实现，而不是共享代码？**
一致性比对的价值来自两条链路的独立性。如果在线只是把离线 SQL 交给 DuckDB 执行，比对就退化成"同一段代码跑两遍"，无法发现任何问题。两套实现 + 逐 cell 比对，才能捕捉到语义漂移。

**为什么缓存到特征级而不是向量级？**
需求原文是"源数据变化时只失效相关的部分"。向量级缓存在任一源变化时整个向量作废；特征级缓存配合依赖图，才能做到 bump `profile` 只影响 3/33 个特征。

**为什么标签要带 `label_ready_time`？**
训练样本的 `as_of` 只约束特征，不约束标签。一个 2025-08 的申请，其 90 天观察窗到 2025-11 才关闭；如果在 2025-09 就用它训练，标签本身是"未来信息"。样本里保留这个时间戳，下游才能按可用性过滤。

**为什么朴素实现不一定提高 AUC？**
"泄漏 = 更好看"是误解。泄漏会让特征值本身错误，AUC 可能上升也可能下降；真正的危害是离线数字与线上不可复现。平台的价值在于用证明 1–4 保证定义正确，而不是优化一个数字。

**已知限制**
- 在线索引全量常驻内存，约 6 万活跃实体的规模下 < 1 GB；更大规模需要分片或换 Redis。
- 数值语义以 DuckDB 为准（如 `CAST` 截断、`arg_max` 跳过 NULL）；若换 Trino/Spark 需重新校准 `expr.py` 的目标语义。
- 漂移监控的参考窗是固定 90 天，未做节假日/季节性调整。
