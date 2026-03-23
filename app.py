import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import pytesseract
import cv2
import os
import re

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("fraud_model.h5")

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Download PDF
@app.route('/download')
def download_file():
    return send_from_directory(".", "report.pdf", as_attachment=True)

# 🔥 Prediction function
def predict_image(path):
    img = Image.open(path).resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Fraud", prediction
    else:
        return "Valid", prediction

# 🔥 PDF Generator
def generate_pdf(result):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Aadhaar Fraud Detection Report", styles['Title']))
    content.append(Paragraph(f"Status: {result['label']}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {result['confidence']}%", styles['Normal']))
    content.append(Paragraph(f"Verdict: {result['verdict']}", styles['Normal']))

    content.append(Paragraph(" ", styles['Normal']))

    content.append(Paragraph(f"Aadhaar: {result['aadhaar']}", styles['Normal']))
    content.append(Paragraph(f"Gender: {result['gender']}", styles['Normal']))
    content.append(Paragraph(f"DOB: {result['dob']}", styles['Normal']))

    if result["reason"]:
        content.append(Paragraph("Fraud Reasons:", styles['Heading2']))
        for r in result["reason"]:
            content.append(Paragraph(f"- {r}", styles['Normal']))

    doc.build(content)

@app.route("/", methods=["GET","POST"])
def index():

    result = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]
        filename = file.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        image_path = filename

        # AI prediction
        label, confidence = predict_image(path)

        # OCR
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        # Extract data
        aadhaar = re.search(r'\d{4}\s?\d{4}\s?\d{4}', text)
        gender = re.search(r'Male|Female', text)
        dob = re.search(r'\d{4}-\d{2}-\d{2}', text)

        # Fraud reasons
        fraud_reason = []

        if not aadhaar:
            fraud_reason.append("Aadhaar Number Missing")

        if "XXXX" in text:
            fraud_reason.append("Masked Aadhaar")

        if confidence > 0.8:
            fraud_reason.append("AI detected suspicious pattern")

        # 🔥 FINAL VERDICT LOGIC
        if label == "Fraud" or len(fraud_reason) > 0:
            verdict = "Suspicious Document"
        else:
            verdict = "Safe Document"

        # Final result
        result = {
            "aadhaar": aadhaar.group() if aadhaar else "Not Found",
            "gender": gender.group() if gender else "Not Found",
            "dob": dob.group() if dob else "Not Found",
            "text": text,
            "label": label,
            "confidence": round(confidence * 100, 2),
            "reason": fraud_reason,
            "verdict": verdict
        }

        # 🔥 GENERATE PDF
        generate_pdf(result)

    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)