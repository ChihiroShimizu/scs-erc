# MELD Stage7R：1,500条标注后的完整验证运行包

本包已合并五个结果包的最终标注、自审与原始盲输入。直接上传本包即可，不需要再上传五个结果 ZIP。

## 实验内容

- 标注：1,500 个唯一目标，来自 866 个训练对话。自审接受 1,500 条，其中保留 1,376 条、重标 124 条；这是内部自审，不是独立人工验证。
- 规则：按原 Stage7 的完整官方 train 对话 ID 划分（90%/10%，seed=2026），必须得到 1,200 条教师训练记录、300 条教师校准记录。服务器会复核输入文本、说话人、轮次及前三轮与现有 MELD 完全一致。比例或文本不一致就停止，不能把标注随机切成1200/300。
- 模型：沿用服务器 `src/pfa_erc/training/llm_evidence_supervision_stage7.py` 的 v2.0.0 API。原文件不会被修改，也不覆盖原实验结果。
- 方法：Qwen2.5-7B-Instruct + 原 LoRA；先预测历史需求及最多一条证据，再判断情绪；推理时隐藏教师标注。
- 种子：42、43、44、45、46。每个种子新训练一个基础情绪 adapter，再独立训练 label-only、evidence、full、shuffled 四个分支，共25个 adapter。
- 主比较：完整额外监督 vs 仅情绪标签继续训练。另比较简化证据监督、打乱监督、分类时隐藏结构字段，并报告同一完整 adapter 的当前句/前三轮/所选证据三种输入。
- 步数：基础情绪训练600；每个新增分支480。各继续训练分支均3600条任务记录、同一回放池；监督 token 数会单独统计，不声称等 FLOPs。
- 边界：只使用 train、train-calibration 和 dev；test 保持封存，不训练自由文本 CoT，不加入强化学习，不作快慢思考或速度声明。

## 1. Mac上传

下载本 ZIP 到 `~/Downloads/`，Mac Terminal 执行：

```bash
scp ~/Downloads/MELD_Stage7R_run1500_v1.zip \
  tyan:/home/chihiro/projects/pfa_erc_reproduction/
```

## 2. 服务器检查和数据准备（CPU）

```bash
ssh tyan
cd ~/projects/pfa_erc_reproduction
unzip -n MELD_Stage7R_run1500_v1.zip
source .venv-server/bin/activate

bash MELD_Stage7R_run1500_v1/run.sh test
bash MELD_Stage7R_run1500_v1/run.sh preflight
bash MELD_Stage7R_run1500_v1/run.sh prepare
```

预期：7项测试通过；preflight 为1500条、1200训练/300校准；prepare 四个继续训练分支均3600条任务记录，并完成 tokenizer 长度审计。

准备阶段会使用项目现有 tokenizer，可能需要读取已有模型缓存。全程不加载GPU模型权重。

如果校验失败，先保留报错，不要手动删记录、改划分或改哈希绕过检查。

本包的 `test` 使用现有核心代码中的数据准备函数做合成数据隔离测试，并检查本批1500条真实标注。缺少原 Stage7 模块、依赖或 API 不兼容时会给出明确错误。本包不自动升级现有训练环境。

## 3. 先做GPU冒烟测试

```bash
nvidia-smi
```

确认物理 GPU4 可用后执行（否则把4换为实际可用编号）：

```bash
GPU_ID=4 bash MELD_Stage7R_run1500_v1/run.sh smoke --seed 42
```

每个 adapter 只训练2步，评估8个目标；包含原方法与新增机制路径。产物位于 `.../meld/smoke_seed42/`，不混入正式结果。成功应显示 `SMOKE PASSED`。

## 4. tmux开始正式实验

```bash
GPU_ID=4 SESSION_NAME=meld7r1500 \
  bash MELD_Stage7R_run1500_v1/launch_tmux.sh run
```

整个任务按固定计划训练全部五个种子，再评估全部种子并生成报告。不是只跑seed42。

进入会话：

```bash
tmux attach -t meld7r1500
```

退出查看界面、保持任务运行：按 `Ctrl-b`，松开后按 `d`。

任务完成或报错后，会话自动关闭，不再停留在空shell。成功退出码为0；失败为非0。日志与退出状态留在 `artifacts/logs/`。

不需要手动开tmux再运行启动脚本，也不要在结束后加 `exec bash` 或 `read`。

## 5. 查看进度和日志

```bash
cd ~/projects/pfa_erc_reproduction
bash MELD_Stage7R_run1500_v1/run.sh status
```

查看正式运行日志可以使用启动时打印的 Log 路径，或使用最近一次GPU任务的快捷路径：

```bash
tail -n 80 artifacts/logs/latest_meld7r1500.log
```

`status`、`report`等CPU命令不会改变这个快捷路径。新的smoke或GPU训练/评估任务会更新它。

退出状态也使用启动时打印的 `.status` 路径：

```bash
cat artifacts/logs/latest_meld7r1500.status
```

文件只有在该命令结束时生成；0为成功。tmux消失但状态非0表示失败，不能视为训练完成。

## 6. 中断后继续

如果任务意外中断，确认旧会话/进程已经退出后，重新执行相同启动命令即可。

完成的 adapter 和整个种子评估在输入、配置和权重一致时会跳过。正在训练但尚未完成的 adapter 会从其固定父 adapter 重新开始该分支，**不声称从中间optimizer step恢复**。

如果只想分种子运行，可以执行：

```bash
GPU_ID=4 bash MELD_Stage7R_run1500_v1/run.sh train-seed --seed 42
GPU_ID=4 bash MELD_Stage7R_run1500_v1/run.sh evaluate-seed --seed 42
```

正式报告仍要求42至46全部完成。相同产物目录有执行锁，不允许同时开启两套写入任务。

## 7. 查看报告并导出

```bash
less artifacts/llm_evidence_supervision_stage7r1500/meld/reports/validation_summary.md

bash MELD_Stage7R_run1500_v1/run.sh export
```

导出成功后，在Mac Terminal下载：

```bash
scp tyan:/home/chihiro/projects/pfa_erc_reproduction/artifacts/llm_evidence_supervision_stage7r1500/meld/MELD_Stage7R_1500_results.zip \
  ~/Downloads/
```

包内含汇总、所有种子分数、每类precision/recall/F1/support、混淆矩阵、纠错/引错统计、300条留出教师记录上的规划诊断，以及数据/配置/代码的冻结哈希。不包含模型权重。

## 实现检查与限制

本地已做：1500条结构及自审一致性检查、实际旧核心数据准备函数的合成隔离测试、配对统计测试、目标错配拒绝测试、Python编译与shell语法检查。

本地未做：真实服务器完整MELD文本/划分校验、tokenizer长度审计、GPU训练或推理。这些由上述 preflight、prepare 和 smoke 在你的服务器完成。不要把本包合成测试的指标当作真实实验结果。

本包提高共享输入预算到1024、规划输出上限到192，以检查完整输入和避免规划JSON被截断；任何实际超长输入都会停止，不会默默截掉历史。与旧500条结果直接作数值差异时，还需说明训练步数、对照记录安排和预算也发生了变化，不能把差异全部归因为标注数量。

400/800/1200学习曲线可作为后续独立扩展；本次默认只执行已约定的1200条监督主消融，不额外启动小规模训练。
