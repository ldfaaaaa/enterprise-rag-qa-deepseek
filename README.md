# 企业文档智能问答系统（Streamlit RAG + DeepSeek-V3）

## 项目背景
这是一个基于RAG的企业知识库问答工具，支持上传文档后智能查询。使用DeepSeek-V3生成回答，带来源引用，防幻觉。适合公司内部知识管理，部署简单，只需一行命令。

## 核心卖点
- 上传文档即用，回答带来源，提升可信度。
- Python本地运行，无需服务器。

## 核心功能
- 上传PDF/TXT文档，自动分块向量化（本地FAISS存储）。
- 基于DeepSeek-V3的智能问答 + 标注来源（文件名 + 段落）。
- 清空知识库按钮（快速重置）。
- Streamlit界面，简单交互。

## 技术栈
- 框架：Streamlit（Python快速Web）。
- RAG：langchain_text_splitters + FAISS向量检索。
- AI模型：DeepSeek-V3。
- 依赖：requirements.txt中列出（numpy, faiss-cpu等）。

## 快速启动
1. 克隆仓库：`git clone https://github.com/你的用户名/enterprise-rag-qa-deepseek-streamlit.git`
2. 安装依赖：`pip install -r requirements.txt`
3. 配置：在app.py中设置DEEPSEEK_API_KEY=你的密钥。
4. 运行：`streamlit run app.py`，浏览器自动打开。
5. 使用：上传PDF/TXT，输入问题查询。
6. 示例：仓库中提供2个测试PDF（下载后上传测试）。

## 演示截图
（插入你的截图）
<img width="3040" height="1636" alt="798c80c6441b0fa7b50b5203f60e2fc0" src="https://github.com/user-attachments/assets/5ff2df6b-b882-4bd5-84f9-c1bd8d858d83" />
<img width="3028" height="1564" alt="373dab4c57910ea7bc4097444d8475bc" src="https://github.com/user-attachments/assets/7b2ee942-d296-4319-8563-445219d61f3f" />
<img width="2974" height="1478" alt="009d3e2f2f8a6149f38ca99242cc12f5" src="https://github.com/user-attachments/assets/eb4c6f75-6aca-4c63-91b8-d5194c5c00dc" />


## 联系方式
想扩展（如支持更多格式或云部署）？联系我，快速定制，7天免费支持。

---
作者：晟（底层程序员，2026年AI副业实践者）
