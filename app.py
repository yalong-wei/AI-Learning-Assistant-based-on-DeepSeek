#!/usr/bin/env python3
"""
AI Learning Assistant - 基于DeepSeek API的智能学习助手
"""

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from openai import OpenAI
from flask.typing import ResponseReturnValue
import mlflow
import mlflow.sklearn
import numpy as np

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 意图分类模型（MLflow加载，按需缓存）
_intent_model = None


def get_intent_model():
    global _intent_model
    if _intent_model is not None:
        return _intent_model
    model_uri = os.getenv("MODEL_URI")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if not model_uri:
        raise RuntimeError(
            "MODEL_URI 未配置，无法加载训练模型。请设置环境变量，如 runs:/<run_id>/model 或 models:/intent-classifier/Production"
        )
    _intent_model = mlflow.sklearn.load_model(model_uri)
    return _intent_model


class DeepSeekClient:
    """DeepSeek API客户端"""

    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY环境变量未设置")

        self.client: Any = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_completion(
        self,
        message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            logger.info("发送请求到DeepSeek API")
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": message},
            ]
            response: Any = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
        except Exception as exc:
            logger.error("DeepSeek API请求失败: %s", exc)
            error_message = str(exc)
            if "timeout" in error_message.lower():
                raise Exception("API请求超时，请稍后重试")
            if "connection" in error_message.lower():
                raise Exception("网络连接失败，请检查网络连接")
            if "authentication" in error_message.lower() or "401" in error_message:
                raise Exception("API密钥无效，请检查DEEPSEEK_API_KEY配置")
            if "rate limit" in error_message.lower() or "429" in error_message:
                raise Exception("API调用频率过高，请稍后重试")
            raise Exception(f"API请求发生错误: {error_message}") from exc

        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens
            if response.usage
            else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        content = response.choices[0].message.content if response.choices else ""
        role = response.choices[0].message.role if response.choices else "assistant"
        return {
            "choices": [{"message": {"role": role, "content": content}}],
            "usage": usage,
        }


try:
    deepseek_client: Optional[DeepSeekClient] = DeepSeekClient()
except ValueError as exc:
    logger.warning("DeepSeek客户端初始化失败: %s", exc)
    deepseek_client = None


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/api/intent/predict", methods=["POST"])
def intent_predict() -> ResponseReturnValue:
    try:
        model = get_intent_model()
    except Exception as exc:
        return jsonify({"error": f"模型不可用: {str(exc)}"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "缺少text参数"}), 400

    text = data["text"]
    # pipeline 期望输入为文本列表
    preds = model.predict([text])
    result = {"label": preds[0]}

    # 若支持 predict_proba，返回前5概率
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba([text])[0]
            # 获取类别标签（按训练时的类顺序）
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
            else:
                classes = []
            topk = sorted(
                zip(classes if classes else range(len(probs)), probs),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            result["topk"] = [{"label": str(c), "prob": float(p)} for c, p in topk]
        except Exception:
            pass

    return jsonify(result), 200


@app.route("/api/chat", methods=["POST"])
def chat() -> ResponseReturnValue:
    if not deepseek_client:
        return jsonify({"error": "DeepSeek API未配置，请检查DEEPSEEK_API_KEY环境变量"}), 500

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "缺少message参数"}), 400

    message = data["message"]
    max_tokens = data.get("max_tokens", os.getenv("MAX_TOKENS", 2048))
    temperature = data.get("temperature", os.getenv("TEMPERATURE", 0.7))

    try:
        response = deepseek_client.chat_completion(
            message=message,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
    except Exception as exc:
        logger.error("聊天处理失败: %s", exc)
        error_message = str(exc)
        if "超时" in error_message:
            return jsonify({"error": "请求超时，请稍后重试"}), 408
        if "网络连接" in error_message:
            return jsonify({"error": "网络连接失败，请检查网络连接"}), 503
        if "API密钥无效" in error_message:
            return jsonify({"error": "API密钥配置错误，请联系管理员"}), 401
        if "频率过高" in error_message:
            return jsonify({"error": "请求频率过高，请稍后重试"}), 429
        return jsonify({"error": f"处理请求时发生错误: {error_message}"}), 500

    if "choices" in response and response["choices"]:
        reply = response["choices"][0]["message"]["content"]
        return jsonify({"reply": reply, "usage": response.get("usage", {})})

    return jsonify({"error": "API响应格式异常"}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream() -> ResponseReturnValue:
    if not deepseek_client:
        return jsonify({"error": "DeepSeek API未配置，请检查DEEPSEEK_API_KEY环境变量"}), 500

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "缺少message参数"}), 400

    message = data["message"]
    max_tokens = data.get("max_tokens", os.getenv("MAX_TOKENS", 2048))
    temperature = data.get("temperature", os.getenv("TEMPERATURE", 0.7))

    @stream_with_context
    def generate():
        yield "event: start\n" + 'data: {"status": "start"}\n\n'
        try:
            stream: Any = deepseek_client.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": message},
                ],
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                stream=True,
            )
            for event in stream:
                delta = getattr(event.choices[0].delta, "content", None)
                if delta:
                    yield "data: " + json.dumps(
                        {"delta": delta}, ensure_ascii=False
                    ) + "\n\n"
            yield "event: end\n" + "data: {}\n\n"
        except Exception as exc:
            err = str(exc)
            yield "event: error\n" + "data: " + json.dumps(
                {"error": err}, ensure_ascii=False
            ) + "\n\n"

    import json

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/health")
def health_check() -> ResponseReturnValue:
    return jsonify(
        {"status": "healthy", "deepseek_configured": deepseek_client is not None}
    )


@app.errorhandler(404)
def not_found(error) -> ResponseReturnValue:
    return jsonify({"error": "资源未找到"}), 404


@app.errorhandler(500)
def internal_error(error) -> ResponseReturnValue:
    return jsonify({"error": "服务器内部错误"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
