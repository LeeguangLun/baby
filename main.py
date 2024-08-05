import cv2
import asyncio
import websockets
import json
import requests
import base64
import os
from dotenv import load_dotenv
import time

# 載入伺服器變數
load_dotenv()
serverIP = os.getenv("SERVER_IP")
serverPort = os.getenv("SERVER_PORT")

ping_url = f"https://db.lyuchan.com/cam_ping?device_id=test1"
alert_url = "https://db.lyuchan.com/alert"
device_id = "test1"

# 開啟相機
cap = cv2.VideoCapture(0)

# 設定相機解析度為1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 檢查相機是否成功開啟
if not cap.isOpened():
    print("錯誤：無法打開攝影機")
    exit()

# 載入人臉偵測模型
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"錯誤：未找到 {cascade_path}")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# 不斷Ping伺服器
async def ping_server():
    while True:
        try:
            response = requests.get(ping_url)
            if response.status_code == 200:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ping 成功：{response.text}")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ping 失敗：{response.status_code}")
        except requests.RequestException as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ping 錯誤：{e}")
        await asyncio.sleep(10)

# 上傳圖片到伺服器並返回圖片連結
def upload_image(frame, picname, replyToken):
    # 將圖片轉換為Base64編碼
    retval, buffer = cv2.imencode('.png', frame)
    base64_data = base64.b64encode(buffer).decode('utf-8')
    url = 'https://db.lyuchan.com/uploadimg'
    payload = {
        'img': base64_data,
        'token': picname,
        'replyToken': replyToken
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print(f"圖片成功上傳：{response.text}")  # 打印伺服器回應
        return response.json().get('img_url')  # 假設伺服器返回的JSON中包含圖片的URL
    except requests.RequestException as e:
        print(f"HTTP 請求失敗：{e}")
        return None

# 處理警報，並包含圖片連結
def send_alert(alertText, img_url=None):
    url = f"{alert_url}?device={device_id}&alertText={alertText}"
    if img_url:
        url += f"&img_url={img_url}"  # 將圖片URL附加到警報信息中
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"成功發送警報：{response.json()}")
    except requests.RequestException as e:
        print(f"無法發送警報：{e}")

# WebSocket客戶端函數
async def websocket_client():
    uri = "wss://db.lyuchan.com"
    async with websockets.connect(uri) as websocket:
        print("已連接到WebSocket伺服器")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if data.get('device') == "test1":
                    # 擷取當前攝影機畫面
                    ret, frame = cap.read()
                    if not ret:
                        print("錯誤：無法擷取影像")
                        continue
                    
                    # 將圖片轉換為灰度圖像
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # 偵測圖像中的人臉
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                    
                    # 保存圖片
                    cv2.imwrite('img.png', frame)
                    print("圖片已本地保存為 img.png")
                    
                    # 上傳圖片到伺服器並獲得圖片連結
                    img_url = upload_image(frame, data.get('picname'), data.get('replyToken'))
                    
                    # 發送警報並附上圖片連結
                    send_alert("未偵測到人臉", img_url)
                    
                elif 'alertText' in data:
                    alertText = data.get('alertText', '未提供警報訊息')
                    send_alert(alertText)
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket連接已關閉")
                break
            except json.JSONDecodeError:
                print("無法解碼JSON訊息")
            except Exception as e:
                print(f"發生錯誤：{e}")

# 主循環函數
async def main():
    # 開始Ping伺服器任務
    ping_task = asyncio.create_task(ping_server())

    # 開始WebSocket客戶端任務
    websocket_task = asyncio.create_task(websocket_client())
    
    last_face_time = time.time()  # 初始化最後偵測到人臉的時間
    last_alert_time = 0  # 初始化最後警報時間
    
    # 顯示攝影機畫面
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法擷取影像")
            break
        
        # 調整畫面尺寸使其更小
        small_frame = cv2.resize(frame, (640, 360))  # 調整此項以改變窗口大小
        
        # 將圖片轉換為灰度圖像
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 偵測圖像中的人臉
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) > 0:
            last_face_time = time.time()  # 更新最後偵測到人臉的時間
        
        # 檢查是否5秒內未偵測到人臉
        if time.time() - last_face_time > 5:
            current_time = time.time()
            if current_time - last_alert_time > 15:  # 檢查距離上次警報是否已經過15秒
                print("異常：未偵測到人臉")
                send_alert("錯誤：未偵測到人臉")
                
                # 擷取並上傳當前畫面，並獲取圖片連結
                ret, alert_frame = cap.read()
                if ret:
                    cv2.imwrite('alert_img.png', alert_frame)
                    print("警報圖片已本地保存為 alert_img.png")
                    img_url = upload_image(alert_frame, 'alert_picname', 'alert_replyToken')
                    
                    # 發送警報並附上圖片連結
                    send_alert("錯誤：未偵測到人臉", img_url)
                
                last_alert_time = current_time  # 更新最後警報時間
            last_face_time = time.time()  # 重設最後偵測到人臉的時間以避免持續打印
        
        # 顯示當前畫面
        cv2.imshow("攝影機", small_frame)
        
        # 按下Esc鍵退出循環
        if cv2.waitKey(1) == 27:
            break
        
        # 允許asyncio執行其他任務
        await asyncio.sleep(0.1)

    # 釋放鏡頭
    cap.release()
    cv2.destroyAllWindows()
    
    # 取消伺服器連接
    ping_task.cancel()
    websocket_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass

# 執行主循環
asyncio.run(main())
