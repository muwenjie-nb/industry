import streamlit as st
from PIL import Image
import requests
import json
import random

st.set_page_config(page_title="设备检修智能系统", page_icon="🔧", layout="wide")

st.title("🔧 基于多模态大模型的设备检修与作业系统")
st.markdown("上传设备故障图片，系统将自动识别缺陷并生成检修方案")

with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key = st.text_input("DeepSeek API Key", type="password", 
                            help="注册 https://platform.deepseek.com/ 获取")
    st.markdown("---")
    st.markdown("### 📋 支持的缺陷类型")
    for defect in [ "夹杂物", "斑点", "麻面", "扎入氧化皮"]:
        st.markdown(f"- 🔴 {defect}")
    st.markdown("---")
    st.markdown("### 📌 使用说明")
    st.markdown("1. 上传设备图片\n2. 点击「开始检测」\n3. 系统自动识别缺陷\n4. 大模型生成检修方案")

def mock_detect():
    """模拟检测结果"""
    defects = [
        {"class": "裂纹 (Crazing)", "confidence": 0.92},
        {"class": "夹杂物 (inclusion)", "confidence": 0.78},
        {"class": "斑点 (patches)", "confidence": 0.85},
        {"class": "麻面 (pitted_surface)", "confidence": 0.71},
        {"class": "扎入氧化皮 (rolled-in_scale)", "confidence": 0.88}
    ]
    num = random.randint(1, 2)
    return random.sample(defects, num)

def call_llm_stream(api_key, defect_type, confidence):
    prompt = f"检测到设备故障：{defect_type}（置信度{confidence:.1%}）。请给出：1.故障原因 2.检修步骤 3.所需工具 4.安全注意事项。"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 600,
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data, stream=True, timeout=60)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                yield content
                        except:
                            pass
        else:
            yield f"API错误: {response.status_code}"
    except Exception as e:
        yield f"请求失败: {e}"

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传设备图片")
    uploaded_file = st.file_uploader("点击上传图片", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        if st.button("🔍 开始检测", type="primary"):
            if not api_key:
                st.error("请输入 API Key")
            else:
                with st.spinner("检测中..."):
                    detections = mock_detect()
                    st.success(f"✅ 检测到 {len(detections)} 处缺陷")
                    with col2:
                        st.subheader("🔍 检测结果")
                        st.image(image, use_container_width=True)
                        for i, d in enumerate(detections):
                            st.info(f"**缺陷 {i+1}**: {d['class']} (置信度: {d['confidence']:.2%})")
                        st.markdown("---")
                        st.subheader("🛠️ 智能检修方案")
                        response = call_llm_stream(api_key, detections[0]['class'], detections[0]['confidence'])
                        st.write_stream(response)

with st.expander("📖 查看示例说明"):
    st.markdown("系统基于 DeepSeek 大模型，支持流式输出检修方案。")
