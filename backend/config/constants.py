# 翻译常量配置

# 翻译提示词模板
TRANSLATION_PROMPT_TEMPLATE = """请你作为具有数十年经验的翻译官，将我下面的文本翻译为我需要的语种。
你需要翻译为：{language}

# 翻译场景
{scene}

# 需要翻译的文本
{text}

{other_requirements}

请你直接输出翻译的结果，不要带其他的文本。
开始翻译："""

# 翻译场景配置
TRANSLATION_SCENES = [
    {
        "id": "ecommerce_amazon",
        "name": "亚马逊电商",
        "description": "适用于亚马逊产品列表、商品描述等电商翻译。注重营销语言和用户体验，保持产品特性的准确传达。"
    },
    {
        "id": "academic",
        "name": "学术文献",
        "description": "适用于学术论文、研究报告等学术文档翻译。注重专业术语的准确性和学术表达的规范性。"
    },
    {
        "id": "finance",
        "name": "金融财务",
        "description": "适用于金融报告、投资文档等财务相关翻译。注重数据的准确性和专业金融术语的使用。"
    },
    {
        "id": "legal",
        "name": "法律文件",
        "description": "适用于合同、法律条款等法律文档翻译。注重法律术语的精确性和条款的严谨性。"
    },
    {
        "id": "technical",
        "name": "技术文档",
        "description": "适用于技术手册、API文档等技术资料翻译。注重技术术语的准确性和操作说明的清晰性。"
    },
    {
        "id": "marketing",
        "name": "营销推广",
        "description": "适用于广告文案、营销材料等推广内容翻译。注重吸引力和说服力，保持品牌调性。"
    },
    {
        "id": "general",
        "name": "通用翻译",
        "description": "适用于日常文档、邮件等一般性内容翻译。保持自然流畅的表达方式。"
    }
]

# 论文翻译提示词模板
PAPER_TRANSLATION_PROMPT_TEMPLATE = """请你作为具有数十年学术翻译经验的专业翻译官，将下面的学术论文片段翻译为中文。
请注意保持学术用词的准确性和专业性，保留原文的格式和结构。

# 需要翻译的论文片段
{text}

请直接输出翻译结果："""

# 支持的语言列表
SUPPORTED_LANGUAGES = [
    {"code": "zh", "name": "中文"},
    {"code": "en", "name": "English"},
    {"code": "es", "name": "Español"},
    {"code": "fr", "name": "Français"},
    {"code": "de", "name": "Deutsch"},
    {"code": "it", "name": "Italiano"},
    {"code": "pt", "name": "Português"},
    {"code": "ru", "name": "Русский"},
    {"code": "ja", "name": "日本語"},
    {"code": "ko", "name": "한국어"},
    {"code": "ar", "name": "العربية"},
    {"code": "hi", "name": "हिन्दी"},
    {"code": "th", "name": "ไทย"},
    {"code": "vi", "name": "Tiếng Việt"},
    {"code": "nl", "name": "Nederlands"},
    {"code": "sv", "name": "Svenska"},
    {"code": "da", "name": "Dansk"},
    {"code": "no", "name": "Norsk"},
    {"code": "fi", "name": "Suomi"},
    {"code": "pl", "name": "Polski"},
    {"code": "tr", "name": "Türkçe"},
    {"code": "he", "name": "עברית"},
]

# 默认模型配置
DEFAULT_MODEL = "doubao-seed-1-6-250615"

# 文件上传配置
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# API配置
API_RATE_LIMIT = 60  # 每分钟请求次数限制