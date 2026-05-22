"""
attendance_deepface.py

Sistema de asistencia para maestros y alumnos usando reconocimiento facial con DeepFace.

Estructura de carpetas esperada:
  dataset/teachers/<nombre>.jpg
  dataset/students/<nombre>.jpg

Requisitos:
  pip install -r requirements.txt

Uso:
  python attendance_deepface.py

Al ejecutar, el sistema abre la cámara, detecta rostros, los compara con la base de datos
persistente y registra la asistencia en attendance.csv.
"""

import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

DATASET_FOLDER = "dataset"
TEACHER_FOLDER = os.path.join(DATASET_FOLDER, "teachers")
STUDENT_FOLDER = os.path.join(DATASET_FOLDER, "students")
ATTENDANCE_FILE = "attendance.csv"
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "mtcnn"
THRESHOLD = 0.45


def create_model():
    return DeepFace.build_model(MODEL_NAME)


def load_reference_embeddings(model):
    labels = []
    embeddings = []

    for role_folder, role_name in [(TEACHER_FOLDER, "Docente"), (STUDENT_FOLDER, "Alumno")]:
        if not os.path.isdir(role_folder):
            continue

        for filename in sorted(os.listdir(role_folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            filepath = os.path.join(role_folder, filename)
            label_name = os.path.splitext(filename)[0].strip()
            label = f"{role_name}:{label_name}"

            try:
                representation = DeepFace.represent(
                    img_path=filepath,
                    model=model,
                    enforce_detection=True,
                    detector_backend=DETECTOR_BACKEND,
                )
            except Exception as e:
                print(f"Error al procesar {filepath}: {e}")
                continue

            if representation and isinstance(representation, list):
                embeddings.append(np.array(representation[0]["embedding"]))
                labels.append(label)

    if len(embeddings) == 0:
        raise RuntimeError(
            "No se encontraron imágenes válidas en dataset/teachers o dataset/students. "
            "Agrega al menos una imagen por persona."
        )

    return np.vstack(embeddings), labels


def init_attendance():
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)

    df = pd.DataFrame(columns=["timestamp", "name", "role", "status"])
    df.to_csv(ATTENDANCE_FILE, index=False)
    return df


def save_attendance(df):
    df.to_csv(ATTENDANCE_FILE, index=False)


def already_registered(df, name, role):
    today = datetime.now().strftime("%Y-%m-%d")
    existing = df[
        (df["name"] == name)
        & (df["role"] == role)
        & (df["timestamp"].str.startswith(today))
    ]
    return not existing.empty


def register_attendance(df, name, role):
    if already_registered(df, name, role):
        return df

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": timestamp, "name": name, "role": role, "status": "Presente"}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_attendance(df)
    print(f"Asistencia registrada: {role} {name} -> {timestamp}")
    return df


def find_best_match(candidate_embedding, reference_embeddings, reference_labels):
    distances = np.linalg.norm(reference_embeddings - candidate_embedding, axis=1)
    best_index = np.argmin(distances)
    best_distance = distances[best_index]

    if best_distance <= THRESHOLD:
        return reference_labels[best_index], best_distance

    return None, None


def parse_label(label):
    role, name = label.split(":", 1)
    return name, role


def run_attendance():
    print("Cargando modelo DeepFace...")
    model = create_model()
    print("Cargando referencias del dataset...")
    reference_embeddings, reference_labels = load_reference_embeddings(model)

    attendance_df = init_attendance()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Verifica que esté conectada.")

    print("Sistema de asistencia iniciado. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                representation = DeepFace.represent(
                    img=face_rgb,
                    model=model,
                    enforce_detection=False,
                    detector_backend=DETECTOR_BACKEND,
                )
            except Exception:
                continue

            if not representation or not isinstance(representation, list):
                continue

            candidate_embedding = np.array(representation[0]["embedding"])
            match_label, distance = find_best_match(candidate_embedding, reference_embeddings, reference_labels)

            if match_label:
                name, role = parse_label(match_label)
                attendance_df = register_attendance(attendance_df, name, role)
                text = f"{role}: {name}"
                color = (0, 255, 0)
            else:
                text = "Desconocido"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Asistencia facial - DeepFace", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_attendance()
