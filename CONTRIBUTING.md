# 贡献指南 (Contributing)

感谢关注与支持 ASRFlow！欢迎提交 Issue 或 Pull Request 来改进本项目。

## 1. 本地开发与运行测试

本项目所有单元测试均基于 Mock 引擎设计，**无需下载模型权重或配置 GPU**，在轻量环境下即可运行。

```bash
# 1. 安装基础依赖
pip install -r requirements-base.txt

# 2. 运行全量单元测试
PYTHONPATH=. python3 -m unittest discover -s tests -v

# 3. 运行单个测试模块示例
PYTHONPATH=. python3 -m unittest tests.test_ring_buffer -v
```

在提交 PR 前，请确保全量测试可以通过：`PYTHONPATH=. python3 -m unittest discover -s tests -v`。

## 2. 提交 Pull Request

1. **清晰简要**：在 PR 中简要说明改动的目的、内容以及本地验证情况。
2. **测试覆盖**：新增功能或修复 Bug 时，请尽量补充或更新对应的单元测试。
3. **保持解耦**：核心网关与 heavy 推理框架（如 vLLM）保持解耦契约，避免引入不必要的强依赖。
4. **敏感信息**：不要在代码或配置文件中硬编码个人凭据、私有地址或 API Key。
