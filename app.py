import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
import json
import os

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="设备检修智能系统",
    page_icon="🔧",
    layout="wide"
)

# ---------- 标题 ----------
st.title("🔧 基于多模态大模型的设备检修与作业系统")
st.markdown("上传设备故障图片，系统将自动识别缺陷并生成检修方案")

# ---------- 侧边栏配置 ----------
with st.sidebar:
    st.header("⚙️ 系统配置")

    # 大模型API配置
    api_key = st.text_input("DeepSeek API Key", type="password",
                            help="注册 https://platform.deepseek.com/ 获取")

    confidence_threshold = st.slider("检测置信度阈值", 0.0, 1.0, 0.25, 0.01)

    st.markdown("---")
    st.markdown("### 📋 支持的缺陷类型")
    st.markdown("- 🔴 裂纹 (Crazing)")
    st.markdown("- 🔴 夹杂物 (inclusion)")
    st.markdown("- 🔴 斑点 (patches)")
    st.markdown("- 🔴 麻面 (pitted_surface)")
    st.markdown("- 🔴 扎入氧化皮 (rolled-in_scale)")
    st.markdown("---")
    st.markdown("### 📌 使用说明")
    st.markdown("1. 上传设备图片")
    st.markdown("2. 点击「开始检测」")
    st.markdown("3. 系统自动识别缺陷")
    st.markdown("4. 大模型生成检修方案（流式输出）")


# ---------- 加载YOLO模型 ----------
@st.cache_resource
def load_model():
    model_path = r"E:\deeplearning\ultralytics-main\runs\detect\result\train11\weights\best.pt"
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在: {model_path}")
        return None
    model = YOLO(model_path)
    return model


model = load_model()

# 类别名称（6类）
class_names = {
    0: "裂纹 (Crazing)",
    1: "夹杂物 (inclusion)",
    2: "斑点 (patches)",
    3: "麻面 (pitted_surface)",
    4: "扎入氧化皮 (rolled-in_scale)"
}


# ---------- 流式调用大模型API ----------
def call_llm_stream(api_key, defect_type, confidence):
    """流式调用DeepSeek API，逐字返回内容"""

    prompt = f"""检测到设备故障：{defect_type}（置信度{confidence:.1%}）。
请给出：1.故障原因 2.检修步骤 3.所需工具 4.安全注意事项。"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位专业的设备检修专家，回答要简洁实用。"},
            {"role": "user", "content": prompt}
        ],
        "stream": True,  # 开启流式输出
        "max_tokens": 800,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True,  # 流式接收
            timeout=90
        )

        if response.status_code == 200:
            # 逐块返回内容
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                        except:
                            pass
        else:
            yield f"API错误: {response.status_code}"

    except Exception as e:
        yield f"请求失败: {e}"


# ---------- YOLO检测函数 ----------
def detect_image(image, model, confidence_threshold=0.25):
    """执行YOLO检测"""
    results = model(image, conf=confidence_threshold)

    detections = []
    result_img = None
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names.get(cls_id, f"未知类别({cls_id})")
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": box.xyxy[0].tolist()
                })
        result_img = r.plot()

    return result_img, detections


# ---------- 主界面布局 ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传设备图片")
    uploaded_file = st.file_uploader("点击上传图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)

        # 检测按钮
        if st.button("🔍 开始检测", type="primary", use_container_width=True):
            if model is None:
                st.error("模型加载失败，请检查模型文件路径")
            elif not api_key:
                st.error("请在侧边栏输入 DeepSeek API Key")
            else:
                with st.spinner("正在检测中..."):
                    # 执行检测
                    result_img, detections = detect_image(image, model, confidence_threshold)

                    if detections:
                        st.success(f"✅ 检测到 {len(detections)} 处缺陷")

                        # 显示检测结果
                        with col2:
                            st.subheader("🔍 检测结果")
                            if result_img is not None:
                                st.image(result_img, caption="检测结果", use_container_width=True)

                            # 显示检测详情
                            for i, det in enumerate(detections):
                                st.info(f"**缺陷 {i + 1}**: {det['class']} (置信度: {det['confidence']:.2%})")

                            # 调用大模型生成检修方案（流式输出）
                            st.markdown("---")
                            st.subheader("🛠️ 智能检修方案")

                            defect_type = detections[0]['class']
                            confidence = detections[0]['confidence']

                            # 流式显示
                            response_stream = call_llm_stream(api_key, defect_type, confidence)
                            st.write_stream(response_stream)  # 一个字一个字蹦出来

                    else:
                        st.warning("⚠️ 未检测到缺陷")
                        with col2:
                            st.info("当前图片未检测到缺陷，请上传包含缺陷的设备图片")
    else:
        st.info("👈 请先上传设备图片")

# ---------- 示例图片 ----------
with st.expander("📖 查看示例说明"):
    st.markdown("""
    ### 使用建议
    1. 本系统可识别裂纹、夹杂物、斑点、麻面、扎入氧化皮等缺陷
    2. 建议上传清晰、光线充足的设备表面图片
    3. 检测完成后，系统会**逐字显示**智能检修方案

    ### 数据来源
    - 检测模型：基于NEU-DET数据集训练（YOLOv8）
    - 大模型：DeepSeek API（流式输出）
    """)