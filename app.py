import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌱 Plant Disease Classifier")
st.markdown("Upload an image of an apple or potato leaf to detect diseases")

# --- Model Definition ---
class DiseaseClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32*32*32, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = DiseaseClassifier(num_classes=4)  # ✅ Match trained model
    model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_model.pth')
    if not os.path.exists(model_path):
        st.error("Model file not found. Please train the model first.")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# --- Class Names ---
CLASS_NAMES = [
    "Apple_healthy",
    "Apple_scab", 
    "Potato_early_blight",
    "Potato_late_blight"  # ✅ Removed "Potato_healthy" or any extra class
]

# --- Upload & Predict ---
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                input_tensor = preprocess_image(image)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    prediction = CLASS_NAMES[predicted.item()]
                
                st.success(f"**Diagnosis:** {prediction.replace('_', ' ')}")
                st.subheader("Confidence Levels")
                for name, prob in zip(CLASS_NAMES, probs):
                    st.write(f"{name.replace('_', ' ')}: {prob*100:.2f}%")
                
                st.subheader("Recommendations")
                if "healthy" in prediction.lower():
                    st.success("✅ The leaf appears healthy!")
                else:
                    st.warning("⚠️ Disease detected.")
                    if "apple" in prediction.lower():
                        st.markdown("- Remove infected leaves\n- Apply fungicide\n- Improve air circulation")
                    elif "potato" in prediction.lower():
                        st.markdown("- Remove infected plants\n- Avoid overhead watering\n- Rotate crops")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Sidebar ---
with st.sidebar:
    st.title("📋 Info")
    st.info("Upload a clear photo of an apple or potato leaf to identify disease.")

    st.title("📸 Tips")
    st.markdown("""
    - Use bright, clear images  
    - Focus on affected parts  
    - Avoid blurry photos
    """)
