# similar_image_retrival

Goal is to search a similar image for a given query image

# Layout
- Use Huggingface dataset: "Matthijs/snacks"
- Train a siamese network using a triplet loss
- Make image data persistent in order to use later in an appropriate storage.


# Dataloader
- Create anchor, positive, images on fly
- make sure to have image augmentation


# Baseline
- Use a pretrained resnet50 model

