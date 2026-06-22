import os
import cv2
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

MODEL_PATH = "final_food_model.keras"
IMG_SIZE = (224, 224)

# Danh sách Class chuẩn từ Model của bạn
CLASS_NAMES = ['ca_hu_kho', 'canh_chua_co_ca', 'canh_chua_khong_ca', 'canh_rau', 'com_trang', 'dau_hu_sot_ca', 'rau_xao', 'suon_nuong', 'thit_kho', 'thit_kho_hot_vit', 'trung_chien']

model = tf.keras.models.load_model(MODEL_PATH)

# 1. ĐỒNG BỘ GIÁ TIỀN (Dùng "trung_chien")
PRICE_DICT = {
    "com_trang": 10000,
    "dau_hu_sot_ca": 25000,
    "ca_hu_kho": 30000,
    "thit_kho_hot_vit": 30000,
    "thit_kho": 25000,          
    "canh_chua_co_ca": 25000,
    "canh_chua_khong_ca": 10000,
    "suon_nuong": 30000,
    "canh_rau": 7000,
    "rau_xao": 10000,
    "trung_chien": 25000            
}

# 2. ĐỒNG BỘ TÙY CHỌN (Đổi key thành "trung_chien" chuẩn theo Model)
OPTIONS_CONFIG = {
    "thit_kho_hot_vit": [
        {"id": "opt_tk_1t", "text": "Mặc định (1 Trứng)", "extra_price": 0},
        {"id": "opt_tk_2t", "text": "Thêm 1 trứng (+6,000đ)", "extra_price": 6000},
        {"id": "opt_tk_3t", "text": "Thêm 2 trứng (+12,000đ)", "extra_price": 12000},
        {"id": "opt_tk_4t", "text": "Thêm 3 trứng (+18,000đ)", "extra_price": 18000}
    ],
    "canh_rau": [
        {"id": "opt_cr_cai", "text": "Rau cải", "extra_price": 0},
        {"id": "opt_cr_muong", "text": "Rau muống", "extra_price": 0}
    ],
    "rau_xao": [
        {"id": "opt_rx_lagim", "text": "Lagim thập cẩm", "extra_price": 0},
        {"id": "opt_rx_san", "text": "Củ sắn", "extra_price": 0},
        {"id": "opt_rx_que", "text": "Đậu que / Đậu đũa", "extra_price": 0}
    ],
    "trung_chien": [
        {"id": "opt_tc_kd", "text": "Mặc định không thịt", "extra_price": 0},
        {"id": "opt_tc_t", "text": "Trứng chiên thịt (+5,000đ)", "extra_price": 5000}
    ]
}

cap = cv2.VideoCapture(1)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/checkout', methods=['POST'])
def checkout():
    data = request.get_json() or {}
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Không kết nối được Camera!"}), 500

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img_rgb.shape
   
    DEFAULT_BOXES = {
        "Box 1": {"x": 50,  "y": 40,  "w": 150, "h": 150},
        "Box 2": {"x": 230, "y": 40,  "w": 150, "h": 150},
        "Box 3": {"x": 410, "y": 40,  "w": 150, "h": 150},
        "Box 4": {"x": 50,  "y": 220, "w": 220, "h": 220},
        "Box 5": {"x": 320, "y": 220, "w": 240, "h": 240}
    }
   
    receipt_items = []
    total_bill = 0
    dynamic_boxes = data.get('boxes', {})
   
    for box_name in ["Box 1", "Box 2", "Box 3", "Box 4", "Box 5"]:
        if box_name in dynamic_boxes and len(dynamic_boxes[box_name]) == 4:
            box_ratio = dynamic_boxes[box_name]
            ymin, xmin, ymax, xmax = [int(box_ratio[0]*img_h), int(box_ratio[1]*img_w), int(box_ratio[2]*img_h), int(box_ratio[3]*img_w)]
        else:
            cfg = DEFAULT_BOXES[box_name]
            x = cfg["x"]
            y = cfg["y"]
            w = cfg["w"]
            h = cfg["h"]
           
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
       
        ymin, xmin, ymax, xmax = max(0, ymin), max(0, xmin), min(img_h, ymax), min(img_w, xmax)
       
        if (ymax - ymin) <= 0 or (xmax - xmin) <= 0:
            continue
           
        crop_img = img_rgb[ymin:ymax, xmin:xmax]
       
        img_resized = cv2.resize(crop_img, IMG_SIZE)
        img_batch = np.expand_dims(img_resized, axis=0)
       
        predictions = model.predict(img_batch, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_label = CLASS_NAMES[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100
       
        if predicted_label != 'o_trong':
            price = PRICE_DICT.get(predicted_label, 0)
           
            display_mapping = {
                "com_trang": "Cơm trắng", 
                "dau_hu_sot_ca": "Đậu hũ sốt cà",
                "ca_hu_kho": "Cá hú kho", 
                "thit_kho_hot_vit": "Thịt kho trứng",
                "thit_kho": "Thịt kho", 
                "canh_chua_co_ca": "Canh chua có cá",
                "canh_chua_khong_ca": "Canh chua không cá", 
                "suon_nuong": "Sườn nướng",
                "canh_rau": "Canh rau", 
                "rau_xao": "Rau xào", 
                "trung_chien": "Trứng chiên"
            }
            
            display_name = display_mapping.get(predicted_label, predicted_label)
            
            # Sửa logic lấy options bám chặt theo nhãn gốc để tránh nhầm lẫn chéo giữa các món
            options = OPTIONS_CONFIG.get(predicted_label, [])
           
            receipt_items.append({
                "vung": box_name,
                "mon": display_name,
                "gia_goc": price,
                "conf": f"{confidence:.1f}",
                "options": options
            })
            total_bill += price
           
    return jsonify({
        "items": receipt_items,
        "total": total_bill
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)