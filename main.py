import cv2
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import logging
import time
import dotenv
import csv
from datetime import datetime
import os

# Load the .env variables
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Telegram bot token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

# IP camera URL (replace with your actual camera URL)
CAMERA_URL = "rtsp://username:password@camera_ip_address"

# Load YOLOv7-tiny model
net = cv2.dnn.readNet("yolov7-tiny.pt", "yolov7-tiny.cfg")

def detect_objects(frame):
    # Convert frame to blob
    blob = cv2.dnn.blobFromImage(frame, 1/255, (640, 480), swapRB=True, crop=False)
    
    # Set input blob for the network
    net.setInput(blob)
    
    # Run object detection
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    
    # Extract bounding boxes and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 0:  # Person
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            
            elif confidence > 0.5 and class_id == 2:  # Car
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def send_telegram_notification(update, context):
    chat_id = update.effective_chat.id
    message = "Object detected in CCTV feed!"
    context.bot.send_message(chat_id=chat_id, text=message)

def save_image_and_update_csv(frame, boxes, confidences, class_ids):
    # Generate a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"detection_{timestamp}.jpg"
    image_path = os.path.join("detections", filename)

    # Save the frame as an image
    cv2.imwrite(image_path, frame)

    # Extract detection information
    detections = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = "Person" if class_ids[i] == 0 else "Car"
        confidence = confidences[i]
        detections.append((label, confidence))

    # Update the CSV file
    csv_file = "detections.csv"
    fieldnames = ["Timestamp", "Image Path", "Detected Objects"]
    with open(csv_file, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow({
            "Timestamp": timestamp,
            "Image Path": image_path,
            "Detected Objects": str(detections)
        })

    print(f"Saved image: {image_path}")
    print(f"Updated CSV file: {csv_file}")

def main():
    # Initialize Telegram bot
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    
    # Add command handler for starting the bot
    dp.add_handler(CommandHandler("start", send_telegram_notification))
    
    # Start the bot
    updater.start_polling()
    
    # Open IP camera stream
    cap = cv2.VideoCapture(CAMERA_URL)
    
    last_detection_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error reading frame")
            continue
        
        boxes, confidences, class_ids = detect_objects(frame)
        
        if len(boxes) > 0:
            # Object(s) detected, save image and update CSV
            save_image_and_update_csv(frame, boxes, confidences, class_ids)
        
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = "Person" if class_ids[i] == 0 else "Car"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            current_time = time.time()
            if current_time - last_detection_time > 60:  # Send notification every minute
                updater.job_queue.run_once(send_telegram_notification, 0, chat_id=update.effective_chat.id)
                last_detection_time = current_time
        
        cv2.imshow('CCTV Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
