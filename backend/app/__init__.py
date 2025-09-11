from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def create_app():
    app = Flask(__name__)
    
    # 配置CORS
    CORS(app, origins=["http://localhost:3000"])
    
    # 配置上传文件夹
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = upload_folder
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
    
    # 注册蓝图
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app