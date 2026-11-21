import os
import cv2
import pickle
import face_recognition

DATASET_DIR = "dataset"
MODEL_DIR = "encodings"

os.makedirs(MODEL_DIR, exist_ok=True)

known_encodings = []
known_names = []
known_rollnos = []

for student_folder in os.listdir(DATASET_DIR):

    student_path = os.path.join(DATASET_DIR, student_folder)

    if not os.path.isdir(student_path):
        continue

    parts = student_folder.split("_", 1)

    if len(parts) != 2:
        print("Invalid folder name:", student_folder)
        continue

    rollno = parts[0]
    name = parts[1]

    print(f"\nProcessing: {rollno} - {name}")

    for filename in os.listdir(student_path):

        image_path = os.path.join(student_path, filename)

        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", filename)
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image)

        if len(face_locations) == 0:
            print("No face found:", filename)
            continue

        if len(face_locations) > 1:
            print("Multiple faces found:", filename)
            continue

        encoding = face_recognition.face_encodings(
            rgb_image,
            face_locations
        )[0]

        known_encodings.append(encoding)
        known_names.append(name)
        known_rollnos.append(rollno)

        print("Added:", filename)

data = {
    "encodings": known_encodings,
    "names": known_names,
    "rollnos": known_rollnos
}

with open("encodings/face_model.pkl", "wb") as file:
    pickle.dump(data, file)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")
print("Total face samples:", len(known_encodings))
print("Model saved at: encodings/face_model.pkl")