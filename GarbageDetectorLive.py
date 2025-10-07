import cv2
import math
import cvzone
import hashlib
import sqlite3
from ultralytics import YOLO

conn = sqlite3.connect("video_hashes.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS video_hashes (hash TEXT PRIMARY KEY)")
conn.commit()


user_conn = sqlite3.connect("users.db")
user_cursor = user_conn.cursor()
user_cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, coins INTEGER)")
user_conn.commit()

def get_video_hash(video_path):
    hasher = hashlib.md5()
    with open(video_path, 'rb') as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_duplicate(video_hash):
    cursor.execute("SELECT hash FROM video_hashes WHERE hash=?", (video_hash,))
    return cursor.fetchone() is not None

def save_hash(video_hash):
    cursor.execute("INSERT INTO video_hashes (hash) VALUES (?)", (video_hash,))
    conn.commit()

def get_user_coins():
    user_cursor.execute("SELECT coins FROM users WHERE id=1")
    result = user_cursor.fetchone()
    return result[0] if result else 0 

def update_user_coins():
    current_coins = get_user_coins()
    new_coins = current_coins + 5 
    user_cursor.execute("INSERT OR REPLACE INTO users (id, coins) VALUES (1, ?)", (new_coins,))
    user_conn.commit()
    print(f"🎉 User rewarded! Total Coins: {new_coins}")

# 🔹 Video path
video_path = "Media/garbage.mp4"

video_hash = get_video_hash(video_path)
print(f"\n🔹 Video Hash: {video_hash}\n")

if is_duplicate(video_hash):
    print("❌ Duplicate video detected! Upload a different video.")
    exit()
else:
    print("✅ New video detected. Processing...")
    save_hash(video_hash)

cap = cv2.VideoCapture(video_path)

model = YOLO("Weights/best.pt")

classNames = ['0', 'c', 'garbage', 'garbage_bag', 'sampah-detection', 'trash']

garbage_detected = False  

while True:
    success, img = cap.read()
    if not success:
        print("✅ Video processing complete.")
        if garbage_detected:
            update_user_coins()  # Reward only if garbage was found
        else:
            print("⚠️ No garbage detected. No coins rewarded.")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            if conf > 0.1 and classNames[cls] in ['garbage', 'garbage_bag', 'trash']:
                garbage_detected = True  # Garbage found at least once
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Garbage Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()