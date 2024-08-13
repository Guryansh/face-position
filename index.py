import os
import csv
from PIL import Image
from landingai.predict import Predictor

endpoint_id = "eb43103c-7b67-40b6-bdcd-dbacd6b434de"
api_key = "land_sk_2hcMnFN5drA3GkdpN3aebIiGSwWNN4pAxK8jVb35wzc7EzlUVr"
image_folder = "google_images"
output_csv = "predictions.csv"

predictor = Predictor(endpoint_id, api_key=api_key)
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][:200]
total_images = len(image_files)

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Score", "Label Name", "Label Index"])

    for count, image_name in enumerate(image_files, start=1):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)

        predictions = predictor.predict(image)

        for prediction in predictions:
            score = prediction.score
            label_name = prediction.label_name
            label_index = prediction.label_index

            writer.writerow([image_name, score, label_name, label_index])
        print(f"Processed {count}/{total_images} images")

print(f"Predictions saved to {output_csv}")
