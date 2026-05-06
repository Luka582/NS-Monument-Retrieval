import torch
from utils import *
import sys
import pandas as pd
import sqlite3
import os
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
register_heif_opener()

model = LandmarkModel(model_name='tf_efficientnetv2_s', num_classes=12015, emb_dim=512)
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu"))["model"])
model.eval()

#theese arguments need to be given when running this script
csv_path = sys.argv[1]
image_folder_path = sys.argv[2]
database_path = sys.argv[3]

df = pd.read_csv(csv_path, sep=",")

with sqlite3.connect(database_path) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE MONUMENT (
            monument_name VARCHAR(255) PRIMARY KEY,
            monument_description TEXT,
            monument_name_serbian VARCHAR(255),
            monument_description_serbian TEXT
        );
        """
    )
    cursor.execute(
        """
        CREATE TABLE VECTOR (
            vector_id INTEGER PRIMARY KEY,
            monument_name VARCHAR(255),
            vector_blob BLOB,
            FOREIGN KEY (monument_name) REFERENCES MONUMENT(monument_name)
        );
        """
    )
    transform = make_inference_transform(384)
    vector_id = 0
    for row in df.itertuples():
        monument_name = row.NAME
        monument_description = row.DESCRIPTION
        monument_name_serbian = row.IME
        monument_description_serbian = row.OPIS
        query = "INSERT INTO MONUMENT (monument_name, monument_description, monument_name_serbian, monument_description_serbian) VALUES (?, ?, ?, ?)"
        cursor.execute(query, (monument_name, monument_description, monument_name_serbian, monument_description_serbian))
        images_path = image_folder_path + "/" + monument_name.replace(" ", "_")
        for image_name in os.listdir(images_path):
            image_adress = images_path + "/" + image_name
            img = Image.open(image_adress).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                vector = model(img)[0]
            vector_blob = vector.numpy().astype('<f4').tobytes()
            query = "INSERT INTO VECTOR (vector_id, monument_name, vector_blob) VALUES (?, ?, ?)"
            value = (vector_id, monument_name, vector_blob)
            vector_id += 1
            cursor.execute(query, value)
            print(f"Image {vector_id-1}")

print("Done.")


