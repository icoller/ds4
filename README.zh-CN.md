# DwarfStar 4 中文说明

[English](README.md) | 中文

## 面向 Agent 的加速版 DS4

这份构建的重点，不只是“能跑 DeepSeek V4 Flash”，而是让它在**真实 coding agent 长任务**里更值得用。

很多官方或通用推理栈在首轮长上下文之后，后续多轮工具调用依然会重复付出很重的 prefill 成本；而这份 DS4 改动的目标，是让已经跑出来的上下文前沿、工具回合状态和可见历史，尽可能继续复用，而不是一轮轮重烧。

如果你在意的是：

- agent 连续几十轮调用工具后，不要越来越慢
- 长上下文撞到上限后，尽量不要前功尽弃
- 已经命中过的前缀、可见历史、tool output 尽量继续复用
- 真实瓶颈要能看见，而不是只看到一个模糊的 “prefill 很慢”

那么这份版本会比单纯“把模型跑起来”的实现更有吸引力。

## 今天这批优化做了什么

### 1. 长任务 continuation 更实用

- `/v1/responses` 支持基于 `previous_response_id` 的 server-managed continuation。
- 当上游 agent 不方便改造时，还可以结合 visible transcript、session hint 和已有前沿做隐式续接。
- `/v1/chat/completions` 也进一步复用了 continuation 思路，不再只有普通文本前缀命中这一条路。

这意味着在 agent 工作流里，超长任务不再只是“撞窗口后报错”，而是更接近**服务端帮你把同一轮工作继续接着跑**。

### 2. tool 回合重复 prefill 明显减少

- chat tool round 现在会优先尝试 live visible-prefix 命中。
- 对工具输出尾部，也会尝试基于 tool-result continuation 直接续到当前 frontier。
- 另外增加了 generic replay frontier，让普通 follow-up chat 也能先走更便宜的复用路径。

简单说，首轮重，后续不该每轮都一样重；这份改动就是在把第二轮、第三轮、后续工具回合的重复成本往下压。

### 3. cold checkpoint 不再双重 prefill

此前为了写 cold KV checkpoint，路径上可能需要：

1. 先 sync 到某个前缀
2. 再 sync 到完整 prompt

今天已经把这块收口成**单次主 sync**：

- 通过一次性的 `prefill boundary`
- 在主 prefill 过程中精准切到目标 frontier
- 当场写入 cold checkpoint

这样做的直接收益是：**长 prompt 下不再为了存一个冷启动前缀，再把同一段前缀额外跑一遍。**

### 4. 少做无意义 CPU 计算

- `prompt anatomy` 统计改成 trace-only 诊断。
- 默认请求不再为了打印拆解日志，额外做多次 tokenization。

这属于很务实的提速：不是让日志更花哨，而是默认路径先别白白消耗 CPU。

### 5. 少做无意义磁盘 I/O

- disk lookup 现在会先检查是否真的存在可复用候选。
- 只有确认可能加载旧 disk checkpoint 替换 live session 时，才先保存当前 live checkpoint。

也就是说，很多“最终根本不会命中磁盘缓存”的请求，现在不会再先写一次盘再发现白忙。

### 6. 日志终于更像性能分析工具

现在的 trace 日志可以直接拆出：

- 本轮总 prompt 多大
- 实际命中了多少 cached token
- 剩余 replay token 大概来自哪里
- 是 recent tail、middle、system 还是 tools 在烧 prefill

这让你看到的不再只是“怎么又慢了”，而是更接近：

> 为什么这轮只命中了前半截？  
> 剩下没命中的 token 到底是工具 schema、system prompt，还是最近几轮消息尾部？

对于调 agent 工作流，这种可见性很关键。

## 这份版本更适合什么场景

- 本地跑 coding agent
- 多轮工具调用
- 长上下文连续会话
- 希望服务端尽量保住已完成工作，而不是上下文一满就直接断掉
- 既在意吞吐，也在意“第二轮开始别越来越慢”

如果你的目标只是单轮 CLI 跑分，这些优化未必最显眼；但如果你的核心诉求是**真实 agent 工作负载下的持续速度与连续性**，这批改动会更有价值。

## 建议阅读路径

- 想看英文原始说明：[`README.md`](README.md)
- 想看速度测试与图表：[`speed-bench/README.md`](speed-bench/README.md)
- 想看量化、imatrix 与模型生成：[`gguf-tools/README.md`](gguf-tools/README.md)
- 想看测试与回归方法：[`CONTRIBUTING.md`](CONTRIBUTING.md)

## 来源

本项目与本文档基于 DS4 进行增强与整理，原始项目见：

- [https://github.com/antirez/ds4](https://github.com/antirez/ds4)
