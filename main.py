from PIL import Image
from src.captioning.captioning import ImageCaptioner
from src.segmentation.segmentation import ImageSegmenter
from src.utils.visualize import visualize

def main():
    image_paths = [
        "data/car.jpeg",
        "data/dog.jpeg",
        "data/person.jpeg"
    ]

    captioner = ImageCaptioner()
    segmenter = ImageSegmenter()

    for image_path in image_paths:
        print("\n" + "="*40)
        print(f"Processing: {image_path}")

        image = Image.open(image_path).convert("RGB")

        caption = captioner.caption(image)
        masks, labels, scores = segmenter.segment(image)

        print("Caption:", caption)
        print(f"Detected {len(labels)} objects.")
        for lbl, sc in zip(labels, scores):
            print(f"Class ID: {lbl}, Confidence: {sc:.2f}")

        visualize(image, masks, labels, caption)

if __name__ == "__main__":
    main()
