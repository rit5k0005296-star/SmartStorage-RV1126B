# main_app.py
import os, cv2, time, sqlite3, threading
import numpy as np
from rknnlite.api import RKNNLite
from math import ceil
from itertools import product as product
from datetime import datetime, timedelta
from flask import Flask, Response, request, render_template, redirect, jsonify, url_for

# 导入你的 YOLO 优化函数及 RKNN 线程池
from rknnpool.rknnpool_ld import rknnPoolExecutor
from func.func_yolov8_optimize import myFunc

app = Flask(__name__)
DB_PATH = 'smart_storage.db'
lock = threading.Lock()
usb_lock = threading.Lock() # 用于保护USB推流图像及状态

YOLO_MODEL_PATH = "./rknnModel/best.rknn"

# --- [1. 加载 RKNN 模型] ---
print("正在加载 RKNN 模型并初始化 NPU...")
det_sess = RKNNLite()
ret1 = det_sess.load_rknn("models/RetinaFace.rknn")
ret2 = det_sess.init_runtime()

rec_sess = RKNNLite()
ret3 = rec_sess.load_rknn("models/MobileFaceNet.rknn")
ret4 = rec_sess.init_runtime()

if ret1 != 0 or ret2 != 0 or ret3 != 0 or ret4 != 0:
    print("NPU 初始化失败！")
    exit(-1)

class GlobalState:
    output_frame = None            # MIPI 人脸图
    usb_output_frame = None        # USB 物品图
    current_counts = {}            # USB当前识别到的物品数量字典
    recognized_user = {"name": "Stranger", "role": "none", "id": None}
    last_seen_time = 0 
    latest_feature = None
    face_db = []

gs = GlobalState()

def load_face_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, role, feature FROM users")
    rows = c.fetchall()
    gs.face_db = [{"id": r[0], "name": r[1], "role": r[2], "feat": np.frombuffer(r[3], dtype=np.float32)} for r in rows]
    conn.close()

# --- [图像处理逻辑] ---
def letterbox_resize(image, size, bg_color):
    target_width, target_height = size
    image_height, image_width, _ = image.shape
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width, new_height = int(image_width * aspect_ratio), int(image_height * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    ox, oy = (target_width - new_width) // 2, (target_height - new_height) // 2
    result_image[oy:oy + new_height, ox:ox + new_width] = image
    return result_image, aspect_ratio, ox, oy

def PriorBox(image_size):
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    f_maps = [[ceil(image_size[0]/s), ceil(image_size[1]/s)] for s in steps]
    for k, f in enumerate(f_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx, s_ky = min_size / image_size[1], min_size / image_size[0]
                cx, cy = (j + 0.5) * steps[k] / image_size[1], (i + 0.5) * steps[k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors).reshape(-1, 4)

def box_decode(loc, priors):
    variances = [0.1, 0.2]
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors):
    variances = [0.1, 0.2]
    return np.concatenate([priors[:, :2] + pre[:, i:i+2] * variances[0] * priors[:, 2:] for i in range(0,10,2)], axis=1)

def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2-x1+1)*(y2-y1+1); order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2-xx1+1), np.maximum(0.0, yy2-yy1+1)
        inter = w*h; ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep

def align_face(img, ldm):
    dst = np.array([[38.29, 51.69], [73.53, 51.50], [56.02, 71.73], [41.54, 92.36], [70.72, 92.20]], dtype=np.float32)
    src = np.array([[ldm[0], ldm[1]], [ldm[2], ldm[3]], [ldm[4], ldm[5]], [ldm[6], ldm[7]], [ldm[8], ldm[9]]], dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(img, tform, (112, 112))

def preprocess_rec(face):
    img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32), axis=0)

# --- [推理核心线程：人脸 (MIPI)] ---
def inference_thread():
    cap = cv2.VideoCapture(23) # 人脸相机保持23
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    import database
    database.init_db()
    load_face_db()
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        lbox, ratio, ox, oy = letterbox_resize(frame, (320, 320), 114)
        outputs = det_sess.inference(inputs=[np.expand_dims(lbox, 0)]) 
        
        loc, conf, landmarks = outputs[0], outputs[1], outputs[2]
        priors = PriorBox(image_size=(320, 320))
        boxes = box_decode(loc.squeeze(0), priors)
        boxes = boxes * np.array([320, 320, 320, 320])
        boxes[..., 0::2] = np.clip((boxes[..., 0::2] - ox) / ratio, 0, frame.shape[1])
        boxes[..., 1::2] = np.clip((boxes[..., 1::2] - oy) / ratio, 0, frame.shape[0])
        
        scores = conf.squeeze(0)[:, 1]
        landms = decode_landm(landmarks.squeeze(0), priors) * np.array([320, 320]*5)
        landms[..., 0::2] = np.clip((landms[..., 0::2] - ox) / ratio, 0, frame.shape[1])
        landms[..., 1::2] = np.clip((landms[..., 1::2] - oy) / ratio, 0, frame.shape[0])
        
        idx = np.where(scores > 0.6)[0]
        dets = np.hstack((boxes[idx], scores[idx, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.4)
        
        temp_user = {"name": "Stranger", "role": "none", "id": None}
        min_dist = 10.0
        
        if len(keep) > 0:
            best_idx = idx[keep[0]]
            face = align_face(frame, landms[best_idx])
            feat = rec_sess.inference(inputs=[preprocess_rec(face)])[0][0]
            feat /= np.linalg.norm(feat)
            gs.latest_feature = feat
            
            for user in gs.face_db:
                dist = np.linalg.norm(feat - user["feat"])
                if dist < min_dist:
                    min_dist = dist
                    if dist < 0.95:
                        temp_user = {"name": user["name"], "role": user["role"], "id": user["id"]}
        
        color = (0, 255, 0) if temp_user["name"] != "Stranger" else (0, 0, 255)
        if len(keep) > 0:
            x1, y1 = int(dets[keep[0]][0]), int(dets[keep[0]][1])
            x2, y2 = int(dets[keep[0]][2]), int(dets[keep[0]][3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{temp_user['name']} ({min_dist:.2f})", (x1, max(y1-10, 10)), 1, 1.5, color, 2)
            
        if temp_user["name"] != "Stranger":
            gs.recognized_user = temp_user
            gs.last_seen_time = time.time()
        else:
            if time.time() - gs.last_seen_time > 2.5: 
                gs.recognized_user = {"name": "Stranger", "role": "none", "id": None}
                
        with lock:
            _, buffer = cv2.imencode('.jpg', frame)
            gs.output_frame = buffer.tobytes()
        time.sleep(0.01)

# --- [推理核心线程：物品 (USB YOLO)] ---
def yolo_inference_thread():
    print("--- [Step 1] 正在探测 USB 摄像头节点 ---")
    test_nodes = [0, 11, 21, 22, 40, 52] # 避开人脸23节点
    cap_usb = None
    for node in test_nodes:
        if node == 23: continue
        temp_cap = cv2.VideoCapture(node)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None:
                print(f"成功获取物品识别图像流: /dev/video{node}")
                cap_usb = temp_cap
                break
        temp_cap.release()
        
    if not cap_usb:
        print("错误: 无法找到可用的 USB 摄像头用于物品识别！")
        return
        
    cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        pool = rknnPoolExecutor(rknnModel=YOLO_MODEL_PATH, TPEs=4, func=myFunc)
    except Exception as e:
        print(f"YOLO RKNN 初始化失败: {e}")
        return

    # 预填充池
    for i in range(5):
        ret, frame = cap_usb.read()
        if ret: pool.put(frame)

    while cap_usb.isOpened():
        ret, frame = cap_usb.read()
        if not ret: continue
        
        pool.put(frame)
        res_frame_and_counts, flag = pool.get()
        
        if flag:
            res_frame, counts = res_frame_and_counts
            display_frame = cv2.resize(res_frame, (640, 480))
            ret_encode, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret_encode:
                with usb_lock:
                    gs.usb_output_frame = buffer.tobytes()
                    gs.current_counts = counts
        else:
            time.sleep(0.01)

# --- [Flask 路由] ---
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with lock:
                if gs.output_frame:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + gs.output_frame + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_usb')
def video_feed_usb():
    def gen_usb():
        while True:
            with usb_lock:
                if gs.usb_output_frame:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + gs.usb_output_frame + b'\r\n')
            time.sleep(0.05)
    return Response(gen_usb(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_current_user')
def get_current_user(): 
    return jsonify(gs.recognized_user)

@app.route('/api/get_counts')
def api_get_counts():
    """获取前端所需要的当前识别数量字典"""
    with usb_lock:
        return jsonify(gs.current_counts)

@app.route('/register_page')
def register_page(): 
    return render_template('register.html')

@app.route('/do_register', methods=['POST'])
def do_register():
    if gs.latest_feature is None: return "请正对摄像头"
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO users (name, role, dorm_info, password, feature) VALUES (?,?,?,?,?)",
                 (request.form['name'], request.form['role'], request.form.get('dorm_info',''), request.form.get('password',''), gs.latest_feature.tobytes()))
    conn.commit(); conn.close(); load_face_db()
    return redirect('/')

@app.route('/dashboard')
def dashboard():
    if not gs.recognized_user["id"]: return redirect('/')
    conn = sqlite3.connect(DB_PATH)
    items = conn.execute("SELECT * FROM items").fetchall()
    my_borrows = conn.execute("SELECT r.id, i.name, r.borrow_time, r.due_time FROM records r JOIN items i ON r.item_id=i.id WHERE r.user_id=? AND r.status='borrowed'", (gs.recognized_user["id"],)).fetchall()
    all_logs = conn.execute("SELECT u.name, i.name, r.borrow_time, r.status FROM records r JOIN users u ON r.user_id=u.id JOIN items i ON r.item_id=i.id").fetchall() if gs.recognized_user["role"] == 'admin' else []
    reports = conn.execute("SELECT r.id, u.name, i.name, r.status, r.admin_note FROM records r JOIN users u ON r.user_id=u.id JOIN items i ON r.item_id=i.id WHERE r.status IN ('damaged', 'lost')").fetchall() if gs.recognized_user["role"] == 'admin' else []
    conn.close()
    return render_template('dashboard.html', user=gs.recognized_user, items=items, my_borrows=my_borrows, all_logs=all_logs, reports=reports)

@app.route('/do_borrow_batch', methods=['POST'])
def do_borrow_batch():
    """接收来自弹窗的批量确认请求并执行借出操作"""
    payload = request.json
    uid = gs.recognized_user['id']
    if not uid: return jsonify({"status": "error", "msg": "用户未识别，请重新登录"})

    conn = sqlite3.connect(DB_PATH)
    try:
        for item_name, count in payload.items():
            count = int(count)
            if count <= 0: continue
            
            row = conn.execute("SELECT id, count FROM items WHERE name=?", (item_name,)).fetchone()
            if not row: continue
            item_id, stock = row
            
            # 以库存为限，避免扣负
            actual_count = min(count, stock)
            
            # 为每一个借出的物品生成一条借还记录
            for _ in range(actual_count):
                due = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
                conn.execute("INSERT INTO records (user_id, item_id, borrow_time, due_time, status) VALUES (?,?,datetime('now','localtime'),?,'borrowed')", (uid, item_id, due))
                conn.execute("UPDATE items SET count = count - 1 WHERE id=?", (item_id,))
        conn.commit()
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})
    finally:
        conn.close()
    return jsonify({"status": "success"})

@app.route('/action/<type>/<int:tid>')
def action(type, tid):
    # 这里保留用于退还/报损的功能
    uid = gs.recognized_user['id']
    conn = sqlite3.connect(DB_PATH)
    if type == 'return':
        iid = conn.execute("SELECT item_id FROM records WHERE id=?", (tid,)).fetchone()[0]
        conn.execute("UPDATE records SET status='returned' WHERE id=?", (tid,))
        conn.execute("UPDATE items SET count = count + 1 WHERE id=?", (iid,))
    elif type in ['damaged', 'lost']:
        conn.execute("UPDATE records SET status=?, admin_note='unprocessed' WHERE id=?", (type, tid))
    conn.commit(); conn.close()
    return redirect('/dashboard')

@app.route('/admin_update_item', methods=['POST'])
def admin_update_item():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE items SET count = count + ? WHERE id=?", (int(request.form['change']), request.form['item_id']))
    conn.commit(); conn.close(); return redirect('/dashboard')

@app.route('/admin_mark_processed/<int:rid>')
def admin_mark_processed(rid):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE records SET admin_note='processed' WHERE id=?", (rid,))
    conn.commit(); conn.close(); return redirect('/dashboard')

if __name__ == '__main__':
    # 启动双线程并行推理
    threading.Thread(target=inference_thread, daemon=True).start()
    threading.Thread(target=yolo_inference_thread, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=False)