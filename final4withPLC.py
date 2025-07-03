from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import time

# PLC Configuration
PLC_IP = '192.168.100.135'
PLC_PORT = 502
DIET_COKE_COIL = 1280  # Coil for Diet Coke detection
ORIGINAL_COKE_COIL = 1281  # Coil for Original Coke detection

# Load YOLOv8 model
model = YOLO(r"C:\Users\pvenn\Desktop\CokeProject\bestt.pt")

# Confidence threshold
CONF_THRESHOLD = 0.3

# Class labels mapping
CLASS_LABELS = {
    0: "Diet Coke",
    1: "Original Coke"
}

# Excel file initialization
excel_file = "detection_counts.xlsx"

# Overwrite the Excel file with a blank DataFrame at the start
df = pd.DataFrame(columns=["Timestamp", "Diet Coke Count", "Original Coke Count"])
df.to_excel(excel_file, index=False)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# PLC Modbus Client
client = ModbusTcpClient(PLC_IP, port=PLC_PORT)

# Try to connect to the PLC
if not client.connect():
    print(f"Failed to connect to PLC at {PLC_IP}:{PLC_PORT}")
    exit()

def toggle_coil(client, coil_address):
    """Activate a coil for a short duration"""
    try:
        # Turn on the coil
        client.write_coil(coil_address, True)
        time.sleep(1)  # Keep it on briefly
        # Turn off the coil
        client.write_coil(coil_address, False)
    except ModbusException as e:
        print(f"Modbus exception: {e}")

# Dictionary to store the latest row data for the current second
latest_data = {}

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run inference
        results = model(frame)

        # Track counts for each class
        original_coke_count = 0
        diet_coke_count = 0

        # Process detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls_id = int(box.cls.item())

            # Map class ID to label
            label = f"{CLASS_LABELS.get(cls_id, 'Unknown')} {conf:.2f}"

            # Filter based on confidence
            if conf >= CONF_THRESHOLD:
                # Count objects and trigger respective coils
                if cls_id == 0:  # Diet Coke
                    diet_coke_count += 1
                    toggle_coil(client, DIET_COKE_COIL)
                elif cls_id == 1:  # Original Coke
                    original_coke_count += 1
                    toggle_coil(client, ORIGINAL_COKE_COIL)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Get the current timestamp trimmed to seconds
        current_second = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update data only if counts are non-zero
        if diet_coke_count > 0 or original_coke_count > 0:
            latest_data[current_second] = {
                "Timestamp": current_second,
                "Diet Coke Count": diet_coke_count,
                "Original Coke Count": original_coke_count
            }

        # Append the latest non-zero data for each second to the DataFrame
        if current_second in latest_data:
            new_data_df = pd.DataFrame([latest_data[current_second]])
            df = pd.concat([df, new_data_df], ignore_index=True)

            # Save the updated DataFrame to Excel
            df.drop_duplicates(subset=["Timestamp"], inplace=True)  # Ensure unique rows per second
            df.to_excel(excel_file, index=False)

            # Clear logged data for the current second to avoid duplicate saves
            latest_data.pop(current_second, None)

        # Display results
        cv2.imshow("YOLOv8 Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    client.close()
def send_email_with_excel(excel_file):
    # Email details
    sender_email = "pvennilah@gmail.com"
    sender_password = "vmsd zrhb cjwr runw"  # Use App Password instead of your Gmail password
    receiver_email = "nandineraja@gmail.com"
    subject = "Coke Classification Detection Counts"

    # Create email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    body = "Please find attached the detection counts Excel sheet."
    message.attach(MIMEText(body, "plain"))

    # Attach the Excel file
    with open(excel_file, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={excel_file}",
        )
        message.attach(part)

    # Send email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)  # Login using App Password
        server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()

# Call email-sending function after program execution
send_email_with_excel(excel_file)


