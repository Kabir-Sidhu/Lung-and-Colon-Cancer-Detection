import tkinter as tk
from tkinter import filedialog
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

class Diagnosis:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer")
        self.model = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer")

        self.root = tk.Tk()
        self.root.title("Lung & Colon Cancer Diagnosis")
        self.root.geometry("400x200")

        self.select_image_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_image_button.pack(pady=20)

        self.root.mainloop()

    def classify_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        predicted_class_idx = logits.argmax(-1).item()

        label_map = {0: "Colon Adenocarcinoma", 1: "Normal colon", 2: "Lung Adenocarcinoma"}
        predicted_label = label_map.get(predicted_class_idx, "Unknown")

        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted class label: {predicted_label}")

        conclusion = self.generate_conclusion(predicted_label)

        self.show_diagnosis(conclusion)

    def generate_conclusion(self, predicted_label):
        if predicted_label == "Lung Adenocarcinoma":
            return (f"Diagnosis: Lung Adenocarcinoma. Lung adenocarcinoma is a type of lung cancer that starts in the glandular cells of the lung tissue."
                    " It is a common form of non-small cell lung cancer (NSCLC), often associated with smoking or exposure to carcinogens. "
                    "Consult a healthcare professional for further evaluation and possible diagnostic tests like CT scans or biopsy.")
        elif predicted_label == "Colon Adenocarcinoma":
            return (f"Diagnosis: Colon Adenocarcinoma. Colon adenocarcinoma is a cancer that originates in the lining of the colon or rectum."
                    " It is the most common type of colorectal cancer and may present with symptoms like changes in bowel habits, blood in stool, or abdominal pain. "
                    "Early detection is crucial for treatment, so further tests like colonoscopy and biopsy are recommended.")
        elif predicted_label == "Normal colon":
            return (f"Diagnosis: Normal colon. Normal colon indicates the tissue appears normal without signs of cancer. However, it's important to maintain regular screening "
                    "for colorectal cancer, especially if there is a family history or other risk factors, as early stages of cancer may not always show clear symptoms.")
        else:
            return "The result is inconclusive. Please consult a healthcare professional for further evaluation."

    def show_diagnosis(self, conclusion):
        diagnosis_window = tk.Toplevel(self.root)
        diagnosis_window.title("Diagnosis")

        diagnosis_window.geometry("400x200")

        label = tk.Label(diagnosis_window, text=conclusion, wraplength=380, justify="center")
        label.pack(pady=20)

    def select_image(self):
        image_path = filedialog.askopenfilename()
        if image_path:
            self.classify_image(image_path)

if __name__ == "__main__":
    app = Diagnosis()