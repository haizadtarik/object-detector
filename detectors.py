from groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T
import cv2
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, OwlViTForObjectDetection, BlipForQuestionAnswering
import numpy as np
import cv2
from typing import List, Tuple, Union
import requests
from torchvision.ops import box_convert


class textDetector:
    def __init__(self, model_name):
        self.model_name = model_name

    def run(self, image_path, text_prompt, threshold):
        if self.model_name == "dino":
            # load model
            model = load_model("./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./GroundingDINO/groundingdino_swint_ogc.pth")
            
            # load image input
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            image = Image.open(image_path).convert("RGB")
            image_transformed, _ = transform(image, None)

            # inference
            boxes, logits, labels = predict(
                model=model,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=threshold,
                text_threshold=0.25,
                device = "cpu"
            )

            w, h = image.size
            boxes = boxes * torch.Tensor([w, h, w, h])
            boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
            logits = logits.tolist()

            # annotate image
            draw = ImageDraw.Draw(image)
            for box, score, label in zip(boxes, logits, labels):
                xmin, ymin, xmax, ymax = box
                draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=4)
                draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="green")

            return boxes, logits, labels, image
    
        elif self.model_name == "owl":
            # load model
            processor = AutoProcessor.from_pretrained("google/owlvit-large-patch14")
            model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")

            # load image input
            image = Image.open(image_path).convert('RGB')
            inputs = processor(text=text_prompt, images=image, return_tensors="pt")

            # inference
            with torch.no_grad():
                outputs = model(**inputs)
                target_sizes = torch.tensor([image.size[::-1]])
                results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
            logits = results["scores"].tolist()
            boxes = results["boxes"].tolist()
            labels = results["labels"].tolist()

            # annotate image
            draw = ImageDraw.Draw(image)
            for box, score, label in zip(boxes, logits, labels):
                xmin, ymin, xmax, ymax = box
                draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=4)
                draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="green")

            return boxes, logits, labels, image
        
        else:
            raise ValueError("Invalid model name. Choose between 'dino' and 'owl'.")


class imageDetector:
    def run(self, image_path, query_image_path):
        processor = AutoProcessor.from_pretrained("google/owlvit-large-patch14")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        image = Image.open(image_path).convert('RGB')
        query_image = Image.open(query_image_path).convert('RGB')
        inputs = processor(images=image, query_images=query_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]
        logits = results["scores"].tolist()
        boxes = results["boxes"].tolist()
        # annotate image
        draw = ImageDraw.Draw(image)
        for box, score in zip(boxes, logits):
            xmin, ymin, xmax, ymax = box
            draw.rectangle((xmin, ymin, xmax, ymax), outline="green", width=4)
            draw.text((xmin, ymin), f"{round(score,2)}", fill="green")
        return boxes, logits, image

    def run_blip(self, target_image_path, query_image_path, threshold, model_name="dino"):
        image = Image.open(query_image_path).convert('RGB')
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
        text = "What is in the picture?"
        inputs = processor(images=image, text=text, return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_answer = processor.decode(outputs[0], skip_special_tokens=True)
        detector = textDetector(model_name)
        return detector.run(image_path=target_image_path, text_prompt=generated_answer, threshold=threshold)
    
