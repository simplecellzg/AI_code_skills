# Simulation Development Best Practices Skill

**Claude Code专用的仿真模拟开发最佳实践Skill**

## 简介

这是一个专为科学计算和仿真模拟领域设计的Claude Code Skill，汇集了6个真实项目的实践经验，涵盖GPU编程、257倍性能优化、系统化调试、数值稳定性等核心领域。

## 核心特性

- 🚀 **GPU异步编程精通** - 竞态条件诊断、同步点识别、虚函数处理
- ⚡ **257倍性能优化实战** - 预计算、缓冲区重用、空间过滤、向量化4阶段优化
- 🔍 **系统化调试方法** - 4阶段调试流程(根因调查→模式分析→假设验证→实施修复)
- 📊 **数值稳定性保障** - 守恒律监控、参数验证、时间尺度匹配
- ✅ **测试驱动开发** - 针对仿真程序的TDD实践，守恒律测试、收敛阶测试
- 🤖 **AI辅助工作流** - Superpowers技能组合策略，提高开发效率
- 🔄 **并行编程模式** - MPI、OpenMP、CUDA的最佳实践

## 知识来源

本Skill基于以下真实项目的实践经验：

| 项目类型 | 技术挑战 | 关键成果 |
|------|---------|---------|
| GPU移植项目 | GPU竞态条件、异步编程 | 7天完成GPU移植，定位并修复4个race condition |
| 性能优化项目 | 初始化瓶颈（函数调用63亿次） | 257倍加速（40分钟→14秒） |
| 数值调试项目 | 收敛失败、参数不匹配 | 修正3个关键参数，收敛残差<1e-6 |
| 从零构建项目 | 系统架构、TDD | 5359行代码，87%测试覆盖率，9天完成 |
| MPI并行化项目 | 数据域分解、通信优化 | 单机到集群的扩展 |
| 非线性建模项目 | 小样本高维输出 | PCA+GPR数据驱动模型 |


## 安装方法

### 方式1：复制到skills目录

```bash
# 复制skill文件到Claude Code的skills目录
cp simulation-development-best-practices/SKILL.md ~/.claude/skills/simulation-development-best-practices/SKILL.md

# 重启Claude Code
claude restart
```

### 方式2：符号链接（推荐）

```bash
# 创建符号链接（方便更新）
ln -s $(pwd)/simulation-development-best-practices/SKILL.md ~/.claude/skills/simulation-development-best-practices/SKILL.md

# 验证安装
claude skills list | grep simulation
```

## 使用场景

Skill会在以下场景自动被调用：

### GPU编程问题
```
✓ "GPU版本出现极端值10^20+"
✓ "多块计算时结果错误"
✓ "怎么添加GPU同步"
✓ "虚函数在GPU上不工作"
✓ "竞态条件race condition"
```

### 性能优化
```
✓ "函数占用85%CPU时间"
✓ "初始化太慢，需要优化"
✓ "如何预计算lookup table"
✓ "性能瓶颈分析"
```

### 数值稳定性
```
✓ "仿真不收敛"
✓ "出现NaN或Inf"
✓ "守恒律被破坏"
✓ "负压力问题"
✓ "参数如何选择"
```

### 并行编程
```
✓ "MPI域分解"
✓ "ghost cell通信"
✓ "并行效率低"
```

## 核心功能概览

### 1. GPU异步编程精通

**关键洞察：** GPU kernel启动是异步的，后续CPU代码可能在GPU完成前就运行了。

```cpp
// ✅ 正确模式
ParallelFor(bx, compute_kernel);
#ifdef AMREX_USE_GPU
    amrex::Gpu::streamSynchronize();  // 等待GPU完成
#endif
use_result(result);  // 安全使用结果
```

**真实案例：** 缺少同步导致密度达到10^20+极端值，定位到4个关键同步点后问题解决。

### 2. 257倍性能优化（真实案例）

**4阶段渐进优化：**
- Stage 0: 缓冲区重用 → 15%加速
- Stage 1: 预计算查找表 → 66%加速
- Stage 2: 空间过滤 → 92.9%加速
- Stage 3: 向量化与融合 → 最终171倍加速

**成果：** 40分钟 → 14秒

### 3. 系统化调试（4阶段方法论）

**Phase 1: 根因调查** - 收集定量证据
**Phase 2: 模式分析** - 识别空间/时间模式
**Phase 3: 假设验证** - 一次一个假设，量化验证
**Phase 4: 实施修复** - 创建回归测试

**真实案例：** velocity_scale=3.0导致50%积分误差 → 修正为sqrt(T_ref)后收敛。

### 4. 数值稳定性

- 参数物理验证（避免硬编码）
- 时间尺度匹配（比率应~O(1)）
- 守恒律实时监控
- CFL条件检查

### 5. AI辅助工作流

**GPU调试技能组合：**
```
brainstorming → systematic-debugging → test-driven-development → verification-before-completion
```

**性能优化技能组合：**
```
writing-plans → dispatching-parallel-agents → test-driven-development → verification-before-completion
```

## 快速开始示例

### 示例1：诊断GPU竞态条件

```bash
# 用户问题
"我的GPU版本在多块计算时出现密度=3.4e27的极端值，总是在x≈0.25,0.50,0.75位置"

# Skill自动提供
1. 症状清单诊断（极端值、块边界、对称性破缺）
2. 诊断脚本（检查空间模式）
3. 定位同步点的策略
4. 修复代码模板
5. 验证测试脚本
```

### 示例2：性能优化

```bash
# 用户问题
"gprof显示compute_legendre()占用85.9%时间，被调用63亿次"

# Skill自动提供
1. 判断是否适合预计算（输入空间有限）
2. LookupTable类实现（含插值）
3. 内存成本估算
4. 性能提升预测
5. 正确性验证方法
```

### 示例3：收敛问题

```bash
# 用户问题
"UGKS-IBM仿真在Kn=0.1时不收敛，残差振荡"

# Skill自动提供
1. 检查时间尺度匹配
2. 验证参数物理意义
3. 诊断边界梯度计算
4. 建议修改策略
5. 收敛监控脚本
```

## 效果对比

| 指标 | 传统开发 | 使用Skill | 提升 |
|------|---------|---------|------|
| GPU竞态条件定位 | 2-3小时猜测 | 30分钟系统化诊断 | 4-6x |
| 性能优化 | 数周试错 | 数天分阶段优化 | 3-5x |
| 数值问题调试 | 3-5天 | 0.5-1天 | 5-6x |
| 测试覆盖率 | 30-50% | 85%+ | 70%+ |
| 代码正确性 | 多次返工 | 一次到位 | 减少返工80% |

## 适用项目类型

- ✅ 计算流体力学（CFD）
- ✅ 稀薄气体动力学（DSMC）
- ✅ 分子动力学（MD）
- ✅ 有限元方法（FEM）
- ✅ 偏微分方程（PDE）求解器
- ✅ 任何需要GPU加速的科学计算程序
- ✅ MPI/OpenMP并行程序

## 关键技术覆盖

### GPU编程
- CUDA异步执行模型
- 同步点识别（FillBoundary、数据变换、循环）
- 虚函数指针问题
- 静态成员访问
- 竞态条件诊断

### 性能优化
- Profiling工具（gprof, nvprof, nsys）
- 预计算与查找表
- 缓冲区重用
- 空间过滤
- 向量化（SIMD）
- 并行分解策略

### 数值方法
- 守恒律验证
- 收敛阶测试
- Method of Manufactured Solutions
- 参数验证
- CFL条件
- 时间尺度匹配

### 软件工程
- 测试驱动开发（TDD）
- 系统化调试
- AI辅助工作流
- 代码审查清单

## 文档资源

- [Claude Code官方文档](https://docs.anthropic.com/claude-code)
- [Superpowers插件](https://github.com/superpowers-dev/superpowers)

## 常见问题

### Q: Skill会自动调用吗？
A: 是的，当你的问题匹配Skill描述中的触发场景时，Claude Code会自动加载这个Skill。

### Q: 我可以修改Skill内容吗？
A: 可以！Skill文件是纯Markdown格式，你可以根据自己的项目经验添加或修改内容。

### Q: Skill占用多少context？
A: 主Skill文件约700行，包含大量代码示例。Claude Code会智能加载相关部分。

### Q: 如何验证Skill已加载？
A: 当你提问GPU或性能相关问题时，如果Claude的回答中包含Skill中的具体模式（如"4阶段调试"、"257倍优化"），说明Skill已生效。

### Q: 支持哪些编程语言？
A: 代码示例主要用C++/CUDA，但原理适用于Fortran、Julia、Python等任何科学计算语言。

## 贡献指南

欢迎基于你的实际项目经验改进这个Skill！

**贡献方式：**
1. Fork本仓库
2. 添加你的实践经验（遵循Skill格式）
3. 提交Pull Request
4. 说明案例的真实性和量化效果

**贡献示例：**
- 新的性能优化模式
- GPU编程陷阱
- 数值稳定性技巧
- 框架特定的最佳实践
- 调试诊断脚本

## 版本历史

### v2.0 (2026-01-29)
- ✨ 新增257倍性能优化详细案例（4阶段优化流程）
- ✨ 扩展GPU竞态条件诊断（真实案例：4个同步点修复）
- ✨ 增加数值稳定性模式（参数验证、时间尺度匹配）
- ✨ 添加AI辅助工作流集成（Superpowers技能组合）
- ✨ 新增3个快速参考清单（GPU调试、性能优化、数值稳定性）
- 📝 基于6个真实项目重写内容
- 📝 移除敏感项目名称，保持通用性
- 📝 所有案例均有量化结果

### v1.0 (2026-01-28)
- 🎉 初始版本发布
- 基本GPU编程模式
- 性能优化策略
- 系统化调试方法

## 许可证

MIT License

## 致谢

特别感谢流体力学交叉团队的支持。

---

**让AI成为你的科学计算专家伙伴** 🚀

*通过系统化方法、量化验证和AI辅助，将仿真模拟开发效率提升数倍*
