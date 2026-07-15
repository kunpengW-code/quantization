# quantization

量化相关技术分析文档归档、小工具开发、vLLM Ascend 量化特性 skill 开发仓库。

## 目录结构

```
quantization/
├── docs/                               # 技术分析文档归档
│   ├── analysis_reports/               # 适配分析报告
│   │   └── vllm_mrv2_quantization_adaptation.md
│   └── skill_dev/                      # skill 开发相关文档
├── tools/                              # 小工具开发（Python）
└── skill_vllm_ascend_quant/           # vLLM Ascend 量化特性 skill
```

## 模块说明

### docs/ - 技术分析文档归档

归档量化相关的技术分析报告、适配分析、调研文档等。

- `analysis_reports/`：特性适配分析报告，例如 vLLM Model Runner V2 量化适配分析、Ascend 量化算子适配分析等。
- `skill_dev/`：skill 开发相关的设计文档、规划文档。

### tools/ - 小工具开发

独立的 Python 小工具，用于辅助量化特性的开发、调试、验证。

每个工具应自包含，可独立运行，并在文件头注明用途与使用方式。

### skill_vllm_ascend_quant/ - vLLM Ascend 量化特性 skill

面向 vLLM Ascend 量化特性场景的 skill 开发目录，遵循 skill 规范。

## 使用约定

- 文档统一使用 Markdown 格式，中文撰写。
- Python 代码遵循 PEP 8，文件头需包含简要用途说明。
- 新增工具或 skill 时，同步更新对应模块的 README。
