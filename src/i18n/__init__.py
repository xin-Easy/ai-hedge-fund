import os
import json
from typing import Dict, Any

# 默认语言设置
DEFAULT_LANGUAGE = 'zh'

# 语言数据缓存
_translations: Dict[str, Dict[str, Any]] = {}


def load_translations(language: str = DEFAULT_LANGUAGE) -> Dict[str, Any]:
    """
    加载指定语言的翻译文件
    """
    if language in _translations:
        return _translations[language]
    
    # 获取i18n目录的路径
    i18n_dir = os.path.dirname(os.path.abspath(__file__))
    lang_dir = os.path.join(i18n_dir, language)
    
    translations = {}
    
    # 遍历语言目录中的所有JSON文件
    if os.path.exists(lang_dir):
        for filename in os.listdir(lang_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(lang_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 使用文件名（不含扩展名）作为命名空间
                    namespace = os.path.splitext(filename)[0]
                    translations[namespace] = json.load(f)
    
    _translations[language] = translations
    return translations


def get_text(key: str, language: str = DEFAULT_LANGUAGE) -> str:
    """
    获取翻译文本
    格式: namespace.path.to.key
    例如: ben_graham.docstrings.agent
    """
    parts = key.split('.')
    if len(parts) < 2:
        return key
    
    namespace = parts[0]
    path = parts[1:]
    
    translations = load_translations(language)
    
    if namespace not in translations:
        return key
    
    current = translations[namespace]
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return key
    
    return current if isinstance(current, str) else key