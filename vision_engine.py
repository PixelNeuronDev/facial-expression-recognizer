import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- CONFIGURATION ---
MODEL_PATH = 'emotion_model.h5'
# Order strictly matched to your training: {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 48  

# 1. Load the Brain
try:
    classifier = load_model(MODEL_PATH)
    print(f"Success: {MODEL_PATH} loaded.")
except Exception as e:
    print(f"Error: Could not load model: {e}")
    exit()

# 2. Load the Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Start Camera
cap = cv2.VideoCapture(0)

print("Camera starting... Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break
    
    frame = cv2.flip(frame, 1) # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw the box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # 1. CROP THE FACE
        roi_gray = gray[y:y+h, x:x+w]
        
        # 2. BOOST CONTRAST (Fixes the "Always Neutral" bug)
        roi_gray = cv2.equalizeHist(roi_gray) 
        
        # 3. RESIZE
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # 4. PREPARE FOR MODEL
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 5. PREDICT
            preds = classifier.predict(roi, verbose=0)[0]
            label = EMOTION_LABELS[preds.argmax()]
            
            # COLOR LOGIC
            if label == 'Happy':
                color = (0, 255, 0) # Green
            elif label == 'Neutral':
                color = (255, 255, 255) # White
            else:
                color = (0, 255, 255) # Yellow

            # 6. DISPLAY RESULT ON FRAME
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # --- THE MISSING PART: SHOW THE WINDOW ---
    cv2.imshow('Emotion Recognition System', frame)

    # --- THE MISSING PART: KEEP WINDOW OPEN ---
    # This waits for 1 millisecond; if 'q' is pressed, it breaks the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up properly
cap.release()
cv2.destroyAllWindows()