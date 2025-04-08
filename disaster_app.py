# -*- coding: utf-8 -*-
# Thêm dòng này ở đầu nếu bạn gặp vấn đề về encoding ký tự tiếng Việt

import tensorflow as tf
import numpy as np
import cv2
import networkx as nx
from geopy.distance import geodesic
import streamlit as st
from PIL import Image
import io
import os
from itertools import combinations

# --- Configuration ---
MODEL_FILENAME = "duong/dan/toi/file/disaster_analysis_model.h5"

CLASS_NAMES = ["Không Thiệt hại", "Thiệt hại Nhẹ", "Thiệt hại Trung bình", "Thiệt hại Nghiêm trọng"]
IMAGE_SIZE = (224, 224)

# --- AI Model Section ---
@st.cache_resource
def load_analysis_model(model_path):
    class MockModel:
        def predict(self, input):
            # Trả về xác suất ngẫu nhiên cho 4 lớp
            probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)), size=1)
            return probs
    return MockModel()
def analyze_image(image_bytes, model):
    if model is None:
        st.error("Mô hình chưa được tải.")
        return None, None
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        pil_resized = pil_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        image_array = np.array(pil_resized) / 255.0
        image_expanded = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_expanded)

        if prediction is not None and len(prediction) > 0:
            prediction_values = prediction[0]
            if len(prediction_values) != len(CLASS_NAMES):
                st.error("Lỗi cấu hình: Số lớp không khớp.")
                return "Lỗi cấu hình", 0.0
            predicted_class_index = np.argmax(prediction_values)
            confidence = float(np.max(prediction_values))
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            return predicted_class_name, confidence
        else:
            st.error("Dự đoán không hợp lệ.")
            return "Lỗi dự đoán", 0.0
    except Exception as e:
        st.error(f"Lỗi phân tích ảnh: {e}")
        return None, None

# --- Route Optimization Section ---
def build_route_graph(all_locations):
    G = nx.Graph()
    if not all_locations or len(all_locations) < 2:
        return G
    valid_nodes = []
    for loc in all_locations:
        if isinstance(loc, tuple) and len(loc) == 2:
            try:
                lat, lon = float(loc[0]), float(loc[1])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    valid_nodes.append((lat, lon))
                    G.add_node((lat, lon))
            except:
                continue
    for loc1, loc2 in combinations(valid_nodes, 2):
        try:
            distance = geodesic(loc1, loc2).km
            if distance > 0:
                G.add_edge(loc1, loc2, weight=distance)
        except:
            continue
    return G

def optimize_route(graph, start, end):
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() < 2:
        return None, 0
    if start not in graph or end not in graph:
        return None, 0
    if start == end:
        return [start], 0
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        distance = nx.shortest_path_length(graph, source=start, target=end, weight='weight')
        return path, distance
    except:
        return None, 0

# --- Streamlit Application ---
st.set_page_config(page_title="AI & Blockchain Quản lý Cứu trợ", layout="wide")
st.title("Ứng dụng AI & Blockchain Quản lý Cứu trợ")

analysis_model = load_analysis_model(MODEL_FILENAME)

col1, col2 = st.columns(2)

with col1:
    st.header("1. Phân tích ảnh khu vực thiên tai")
    uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh đã tải lên", width=350)
        if analysis_model is not None:
            if st.button("Phân tích Mức độ Thiệt hại"):
                with st.spinner("Đang phân tích ảnh..."):
                    image_bytes = uploaded_file.getvalue()
                    predicted_label, confidence = analyze_image(image_bytes, analysis_model)
                    if predicted_label is not None:
                        st.success("Phân tích hoàn tất!")
                        st.metric(label="Mức độ thiệt hại", value=predicted_label)
                        st.metric(label="Độ tin cậy", value=f"{confidence:.2%}")
                        st.write("Model loaded:", analysis_model is not None)


with col2:
    st.header("2. Tối ưu tuyến đường cứu trợ")
    intermediate_locations_str = st.text_area("Địa điểm trung gian", "10.8231, 106.6297\n10.7625, 106.6826")
    start_location_str = st.text_input("Điểm xuất phát")
    end_location_str = st.text_input("Điểm đến")

    if st.button("Tính toán tuyến đường tối ưu"):
        try:
            start = tuple(map(float, start_location_str.strip().split(',')))
            end = tuple(map(float, end_location_str.strip().split(',')))
            intermediates = [tuple(map(float, line.strip().split(','))) for line in intermediate_locations_str.strip().split('\n') if line.strip()]
            all_points = [start] + intermediates + [end]
            graph = build_route_graph(all_points)
            path, distance = optimize_route(graph, start, end)
            if path:
                st.success("Tuyến đường tối ưu:")
                st.write(" ➔ ".join([f"({p[0]:.4f},{p[1]:.4f})" for p in path]))
                st.metric("Tổng khoảng cách", f"{distance:.2f} km")
            else:
                st.error("Không tìm được tuyến đường phù hợp.")
        except:
            st.error("Định dạng tọa độ sai.")

import hashlib
import time

# --- Blockchain Section ---

@st.cache_data(show_spinner=False)
def initialize_blockchain():
    """Khởi tạo blockchain với block gốc (genesis)."""
    genesis_block = {
        "index": 0,
        "timestamp": time.time(),
        "data": "Khởi tạo hệ thống cứu trợ",
        "previous_hash": "0",
        "hash": ""
    }
    genesis_block["hash"] = hash_block(genesis_block)
    return [genesis_block]

def hash_block(block):
    """Tạo mã hash SHA-256 cho block."""
    block_string = f"{block['index']}{block['timestamp']}{block['data']}{block['previous_hash']}"
    return hashlib.sha256(block_string.encode()).hexdigest()

def add_block(blockchain, data):
    """Thêm block mới vào chuỗi."""
    last_block = blockchain[-1]
    new_block = {
        "index": len(blockchain),
        "timestamp": time.time(),
        "data": data,
        "previous_hash": last_block["hash"],
        "hash": ""
    }
    new_block["hash"] = hash_block(new_block)
    blockchain.append(new_block)

# Sử dụng session_state để giữ blockchain giữa các lần submit
if "blockchain" not in st.session_state:
    st.session_state.blockchain = initialize_blockchain()

# --- Giao diện nhập dữ liệu cứu trợ ---
st.subheader("Ghi nhận hàng cứu trợ")

colA, colB = st.columns(2)
with colA:
    item_name = st.text_input("Tên hàng hóa", key="item_name")
    quantity = st.number_input("Số lượng", min_value=1, step=1, key="quantity")
with colB:
    location = st.text_input("Địa điểm phân phát", key="relief_location")
    receiver = st.text_input("Người nhận", key="receiver")

if st.button("Ghi nhận giao dịch"):
    if item_name and location and receiver:
        data = f"Hàng: {item_name}, Số lượng: {quantity}, Địa điểm: {location}, Người nhận: {receiver}"
        add_block(st.session_state.blockchain, data)
        st.success("Đã ghi nhận vào blockchain!")
    else:
        st.warning("⚠️ Vui lòng điền đầy đủ thông tin trước khi ghi nhận.")

# --- Truy xuất dữ liệu ---
st.subheader("Lịch sử giao dịch (Blockchain)")

for block in st.session_state.blockchain:
    with st.expander(f"Block #{block['index']} - {time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(block['timestamp']))}"):
        st.write(f"**Dữ liệu:** {block['data']}")
        st.write(f"**Hash:** `{block['hash']}`")
        st.write(f"**Hash trước:** `{block['previous_hash']}`")
