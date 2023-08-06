# Text Guided or Image Guided Object Detection

This repository contains the code for implementing text-guided object detection and image-guided object detection.

The text-guided object detector is implmented using [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) from Huggingface Transformers.

The image-guided object detector is implemented using [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) from Huggingface Transformers and 
integrated with the text-guided object detector by using [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) from Huggingface Transfomrers to get text prompt


## Setup
1. Clone the repository
    ```
    git clone https://github.com/haizadtarik/object-detector.git
    ```

2. Install requirements
    ```
    pip install -r requirements.txt
    ```

2. Install and setup GroundingDino
    ```
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO/
    pip install -e .
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    cd ..
    ```

# Example Usage

Text Guided with GroundingDINO
    ```
    from detectors import textDetector
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = requests.get(query_url, stream=True).raw
    text = "cat"
    detector = textDetector("dino")
    boxes, logits, labels, image_target = detector.run(image_path=image_path, text_prompt=text, threshold=0.2)
    image_target.show()
    ```

Text Guided with OWL-ViT
    ```
    from detectors import textDetector
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = requests.get(query_url, stream=True).raw
    text = "cat"
    detector = textDetector("owl")
    boxes, logits, labels, image_target = detector.run(image_path=image_path, text_prompt=text, threshold=0.2)
    image_target.show()
    ```

Image Guided with OWL-ViT
    ```
    from detectors import imageDetector
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_path = requests.get(query_url, stream=True).raw

    query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
    query_image_path = Image.open(requests.get(query_url, stream=True).raw)

    detector = imageDetector()
    boxes, logits, labels, image_target = detector.run(image_path=image_path, query_image_path=query_image_path)
    image_target.show()
    ```

Image guided with blip and dino
    ```
    from detectors import imageDetector
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    target_image_path = requests.get(query_url, stream=True).raw

    query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
    query_image_path = Image.open(requests.get(query_url, stream=True).raw)

    detector = imageDetector()
    boxes, logits, labels, image_target = detector.run(target_image_path, query_image_path, threshold=0.2, model_name="dino")
    image_target.show()
    ```


