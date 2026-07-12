from flask import Flask, render_template, request
from ultralytics import YOLO
import easyocr
import cv2
import os
import re
import pymysql
import boto3

# Flask App
app = Flask(
    __name__,
    template_folder='../templates'
)

# Load YOLO Model
model = YOLO("runs/detect/train-3/weights/best.pt")

# EasyOCR Reader
reader = easyocr.Reader(['en'])

# Upload Folder
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# RDS Database Configuration
db = pymysql.connect(
    host="anpr-db.cr86yc6wwwl5.ap-south-1.rds.amazonaws.com",
    user="admin",
    password="admin0000",
    database="anpr_system"
)

cursor = db.cursor()

# S3 Configuration
s3 = boto3.client(
    "s3",
    region_name="ap-south-1"
)

BUCKET_NAME = "anpr-vehicle-images-ashwin5304"

sns = boto3.client(
    "sns",
    region_name="ap-south-1"
)

TOPIC_ARN = "arn:aws:sns:ap-south-1:568890175566:anpr-alerts"

# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# HISTORY PAGE
@app.route('/history')
def history():

    cursor.execute(
        """
        SELECT
            id,
            plate_number,
            image_name,
            image_url,
            detected_time
        FROM vehicle_logs
        ORDER BY detected_time DESC
        """
    )

    records = cursor.fetchall()

    return render_template(
        'history.html',
        records=records
    )

# DETECTION ROUTE
@app.route('/detect', methods=['POST'])
def detect():

    detected_numbers = []

    # Get uploaded file
    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Save uploaded image locally
    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

        # Upload image to S3
    try:
        s3.upload_file(
            filepath,
            BUCKET_NAME,
            file.filename
        )

        # Generate Pre-Signed URL
        image_url = s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': file.filename
            },
            ExpiresIn=86400
        )

        print("S3 URL:", image_url)

    except Exception as e:
        print("S3 Upload Error:", e)
        image_url = None

    # YOLO Detection
    results = model(filepath, imgsz=960)

    # Read image
    image = cv2.imread(filepath)

    # Process detections
    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # Crop plate
            plate_crop = image[y1:y2, x1:x2]

            # Skip invalid crop
            if plate_crop.size == 0:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(
                plate_crop,
                cv2.COLOR_BGR2GRAY
            )

            # Resize for better OCR
            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2
            )

            # Noise reduction
            gray = cv2.bilateralFilter(
                gray,
                11,
                17,
                17
            )

            # Sharpen image
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3)
            )

            gray = cv2.morphologyEx(
                gray,
                cv2.MORPH_CLOSE,
                kernel
            )

            # Thresholding
            gray = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]

            # OCR
            ocr_result = reader.readtext(gray)

            plate_text = ""

            for detection in ocr_result:

                text = detection[1]
                confidence = detection[2]

                print(text, confidence)

                # Ignore only IND
                if text.upper() != "IND":
                    plate_text += text + " "

            # Clean text
            plate_text = plate_text.replace(" ", "")
            plate_text = plate_text.upper()

            # Keep only letters and numbers
            plate_text = re.sub(
                r'[^A-Z0-9]',
                '',
                plate_text
            )

            # OCR corrections
            plate_text = plate_text.replace("Z", "2")
            plate_text = plate_text.replace("O", "0")
            plate_text = plate_text.replace("Q", "0")
            plate_text = plate_text.replace("I", "1")
            plate_text = plate_text.replace("L", "1")
            plate_text = plate_text.replace("S", "5")
            plate_text = plate_text.replace("B", "8")
            plate_text = plate_text.replace("G", "6")

            # Limit length
            plate_text = plate_text[:10]

            print("Detected:", plate_text)

            # Add only valid detections
            if (
                plate_text != ""
                and plate_text not in detected_numbers
            ):

                detected_numbers.append(plate_text)
                                # Blacklisted vehicles
                                # Blacklisted vehicles
                BLACKLIST = [
                    "KA20HE5304",
                    "KA35N4140"
                ]

                if plate_text in BLACKLIST:

                    try:

                        sns.publish(
                            TopicArn=TOPIC_ARN,
                            Subject="ANPR Alert - Blacklisted Vehicle Detected",
                            Message=f"""
Vehicle Detected

Plate Number: {plate_text}

Image: {image_url}

Time: Detection recorded in ANPR system.
"""
                        )

                        print("SNS Alert Sent")

                    except Exception as e:

                        print("SNS Error:", e)
                try:
                    cursor.execute(
                        """
                        INSERT INTO vehicle_logs
                        (plate_number, image_name, image_url)
                        VALUES (%s, %s, %s)
                        """,
                        (
                            plate_text,
                            file.filename,
                            image_url
                        )
                    )

                    db.commit()

                except Exception as e:
                    print("Database Insert Error:", e)

    print(detected_numbers)

    return render_template(
        'index.html',
        detected_numbers=detected_numbers,
        uploaded_image=file.filename
    )


# RUN FLASK
if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000
    )