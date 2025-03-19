import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

model = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(img):
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  
        transforms.ToTensor(),           
        transforms.Normalize(            
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    ])

    img_tensor = transform(img).unsqueeze(0)  

    with torch.no_grad():
        embedding = model(img_tensor)

    return embedding.numpy() 

# image_path = "./Face_detect/pr.png"
# img = Image.open(image_path)
# embeddings = get_face_embedding(img)

# print("Embeddings shape:", embeddings.shape)