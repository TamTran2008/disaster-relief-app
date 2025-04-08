import tensorflow as tf
import numpy as np
import cv2
import networkx as nx
from geopy.distance import geodesic
import streamlit as st

# AI Model: Phân tích ảnh khu vực thiên tai
def load_model():
    model = tf.keras.models.load_model("disaster_analysis_model.h5")
    return model

def analyze_image(image, model):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction  # Trả về xác suất mức độ thiệt hại

# Tối ưu tuyến đường bằng thuật toán Dijkstra
def optimize_route(locations, start, end):
    G = nx.Graph()
    for loc1, loc2 in zip(locations[:-1], locations[1:]):
        distance = geodesic(loc1, loc2).km
        G.add_edge(loc1, loc2, weight=distance)
    
    shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
    return shortest_path

# Ứng dụng giao diện
st.title("Ứng dụng AI & Blockchain Quản lý Cứu trợ")

st.header("Phân tích ảnh khu vực thiên tai")
uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    model = load_model()
    damage_level = analyze_image(image, model)
    st.write(f"Mức độ thiệt hại dự đoán: {damage_level}")

st.header("Tối ưu tuyến đường cứu trợ")
start_location = st.text_input("Điểm xuất phát (lat, lon)")
end_location = st.text_input("Điểm đến (lat, lon)")
if st.button("Tính toán tuyến đường tối ưu"):
    locations = [(10.7769,106.7009), (10.8231,106.6297), (10.7625,106.6826)]
    if start_location and end_location:
        start = tuple(map(float, start_location.split(',')))
        end = tuple(map(float, end_location.split(',')))
        optimal_path = optimize_route(locations, start, end)
        st.write(f"Tuyến đường tối ưu: {optimal_path}")
