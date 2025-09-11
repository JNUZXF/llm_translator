# AI翻译助手
一个基于React + Flask的小清新风格AI翻译网站，支持快速文本翻译和PDF文档翻译。

## 功能特性

### 🚀 Fast Translation（快速翻译）
- 实时文本翻译
- 支持22种主流语言
- 流式输出，翻译过程可视化
- 防抖输入，自动触发翻译

### 📄 Paper Translation（论文翻译）
- PDF文件上传和解析
- 按页面分段翻译
- 上传后自动触发整份PDF的并行流式翻译（SSE），实时显示结果
- 学术论文专用翻译模型
- 原文与译文对照显示

### 🎨 界面设计
- 小清新渐变背景
- 动态浮动花朵动画
- 毛玻璃质感设计
- 响应式布局
 
## 技术栈

### 前端
- React 18 + TypeScript
- Styled Components
- Server-Sent Events (SSE)

### 后端
- Flask + Python
- 支持多种大模型API
- 流式响应处理
- PDF文档解析

### 支持的大模型
- 豆包系列模型
- OpenAI GPT系列
- Claude系列
- Gemini系列
- DeepSeek系列
- 更多模型...

## 快速开始

### 1. 环境要求
- Python 3.8+
- Node.js 16+
- npm 或 yarn

### 2. 安装依赖
双击运行 `install_dependencies.bat` 自动安装所有依赖

或手动安装：
```bash
# 创建虚拟环境
python -m venv venv
# 激活虚拟环境
source venv/bin/activate
# windows
venv\Scripts\activate

cd ai-translator
# 安装后端依赖
cd backend
pip install -r requirements.txt

# 安装前端依赖
cd ../frontend
npm install
```

### 3. 配置环境变量
在后端目录创建 `.env` 文件，配置API密钥：
```env
# 豆包API配置
DOUBAO_API_KEY=your_doubao_api_key

# OpenAI API配置  
OPENAI_API_KEY=your_openai_api_key

# 其他模型API配置...
```

### 4. 启动应用
双击运行 `start_app.bat` 一键启动应用

或手动启动：
```bash
# 启动后端服务
cd backend
python app.py

# 启动前端服务
cd frontend
npm start
```

### 5. 访问应用
打开浏览器访问：http://localhost:3000

## 项目结构

```
ai-translator/
├── backend/                 # Flask后端
│   ├── app/                # 应用主体
│   │   ├── __init__.py    # Flask应用初始化
│   │   └── routes.py      # API路由
│   ├── config/             # 配置文件
│   │   └── constants.py   # 常量和提示词
│   ├── utils/              # 工具模块
│   │   └── pdf_processor.py # PDF处理工具
│   ├── tools_agent/        # LLM管理模块
│   ├── app.py             # 应用入口
│   └── requirements.txt   # Python依赖
├── frontend/               # React前端
│   ├── src/
│   │   ├── components/    # React组件
│   │   ├── pages/         # 页面组件
│   │   ├── styles/        # 样式文件
│   │   ├── utils/         # 工具函数
│   │   ├── constants/     # 常量定义
│   │   └── types/         # TypeScript类型
│   ├── public/            # 静态资源
│   └── package.json       # Node.js依赖
├── start_app.bat          # 一键启动脚本
├── install_dependencies.bat # 依赖安装脚本
└── README.md              # 项目说明
```

## API接口

### 翻译相关
- `POST /api/translate` - 快速文本翻译
- `POST /api/translate-paper` - 论文翻译
- `POST /api/translate-pdf` - 整份PDF并行流式翻译（输入参数：`filepath` 为 `/api/upload` 返回的文件路径）
- `GET /api/languages` - 获取支持的语言列表

### 文件处理
- `POST /api/upload` - 上传PDF文件

### 系统
- `GET /api/health` - 健康检查

## 开发说明

### 添加新的语言支持
在 `backend/config/constants.py` 和 `frontend/src/constants/index.ts` 中添加新语言配置。

### 集成新的大模型
在 `backend/tools_agent/llm_manager.py` 中添加新的Provider类。
### 整份PDF流式翻译（SSE）说明
- 后端新增 `POST /api/translate-pdf`，内部使用 `utils/agent_tool_pdf_translation.py` 的 `AsyncPDFTranslator` 调用 OCR→Markdown→分段→并行翻译，按顺序流式输出。
- 前端在 `PaperTranslation` 中：上传成功后自动调用 `api.translatePDF(filepath)`，监听流式结果并累加到结果区。
- 如果需要取消，可调用现有 `POST /api/cancel-translation`，传入会话ID（接口会在流开始时先推送 `session_id`）。


### 自定义样式
修改 `frontend/src/styles/GlobalStyles.ts` 中的样式组件。

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！