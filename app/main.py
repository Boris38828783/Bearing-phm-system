"""
轴承故障诊断 API 服务
"""
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ---------- 配置 ----------
MODEL_PATH = "models/cnn1d_noise_aug.onnx"
CLASS_NAMES = ["正常", "内圈故障", "外圈故障"]
INPUT_LENGTH = 2048

# ---------- 初始化 ----------
app = FastAPI(
    title="轴承故障诊断 API",
    description="基于 1D-CNN 的滚动轴承振动信号故障诊断服务",
    version="1.0.0",
)

# 服务启动时加载一次模型（不要每次请求都加载！）
session = ort.InferenceSession(MODEL_PATH)


# ---------- 请求/响应的数据结构 ----------
class SignalRequest(BaseModel):
    """输入：一条或多条振动信号"""
    signals: List[List[float]]

    class Config:
        json_schema_extra = {
            "example": {"signals": [[0.01, -0.02, 0.03]]}
        }


class DiagnosisResult(BaseModel):
    """输出：单条信号的诊断结果"""
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: dict


# ---------- 工具函数 ----------
def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Z-score 归一化（与训练时保持一致）"""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def softmax(x: np.ndarray) -> np.ndarray:
    """把 logits 转成概率"""
    e = np.exp(x - x.max(axis=1, keepdims=True))   # 减最大值防溢出
    return e / e.sum(axis=1, keepdims=True)


# ---------- 接口 ----------
@app.get("/")
def root():
    """服务状态检查"""
    return {
        "service": "轴承故障诊断 API",
        "status": "running",
        "model": MODEL_PATH,
        "input_length": INPUT_LENGTH,
        "classes": CLASS_NAMES,
    }


@app.get("/health")
def health():
    """健康检查接口（部署时用于探活）"""
    return {"status": "healthy"}


@app.post("/predict", response_model=List[DiagnosisResult])
def predict(request: SignalRequest):
    """
    对振动信号进行故障诊断。

    输入：signals - 二维数组，每行是一条长度为 2048 的振动信号
    输出：每条信号的预测类别、置信度和各类概率
    """
    x = np.array(request.signals, dtype=np.float32)

    # --- 输入校验（生产级 API 必备）---
    if x.ndim != 2:
        raise HTTPException(400, f"输入应为二维数组，收到 {x.ndim} 维")
    if x.shape[1] != INPUT_LENGTH:
        raise HTTPException(
            400, f"信号长度应为 {INPUT_LENGTH}，收到 {x.shape[1]}"
        )

    # --- 预处理：必须和训练时完全一致 ---
    x = normalize_signal(x)
    x = x[:, np.newaxis, :]          # (N, 2048) -> (N, 1, 2048)

    # --- 推理 ---
    logits = session.run(["logits"], {"signal": x})[0]
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    # --- 组装结果 ---
    results = []
    for i, p in enumerate(preds):
        results.append(DiagnosisResult(
            predicted_class=int(p),
            class_name=CLASS_NAMES[p],
            confidence=float(probs[i, p]),
            probabilities={
                name: float(probs[i, j]) for j, name in enumerate(CLASS_NAMES)
            },
        ))
    return results