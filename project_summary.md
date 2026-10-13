# Compression-Robust Deepfake Detection: Project Summary

## 1. Project Overview
The core objective of this project was to build a deepfake detection system capable of identifying AI-generated media even after it has been subjected to severe, aggressive compression—specifically the proprietary compression algorithms used by platforms like WhatsApp. Traditional deepfake detectors rely heavily on high-frequency noise and pixel-level artifacts, which are entirely destroyed by WhatsApp's compression, causing standard models to fail.

## 2. Datasets Used
To train, evaluate, and specialize the model, we utilized three distinct data splits:

*   **Global Image Dataset (`facefake`):** A massive dataset of 10,000 images (5,000 Real, 5,000 Fake). We artificially applied JPEG compression to these images at various Quality Factors (Clean, QF90, QF70, QF50, QF30) to simulate standard web compression.
*   **Global Video Dataset (`deepfake-video`):** A dataset of 400 video clips (200 Real, 200 Fake) used to train the temporal (time-based) detection layers.
*   **Custom WhatsApp Dataset (User Provided):** A highly curated, specialized dataset of 28 heavily compressed WhatsApp files (Images: 13 Real, 7 Fake | Videos: 5 Real, 3 Fake). This dataset was crucial for fine-tuning the model to understand the specific destructive footprint of WhatsApp.

## 3. Algorithms & Architecture

### Image Detection (Spatial Analysis)
We moved away from traditional Convolutional Neural Networks (CNNs) like EfficientNet because they suffered from "semantic bias" (e.g., assuming all images in a certain environment were real). 

*   **CLIP ViT-L/14 Backbone:** We used the Vision Transformer (ViT) from OpenAI's CLIP model. CLIP is incredibly robust at understanding high-level image semantics. By unfreezing the **last 6 layers** of this model, we forced it to learn domain-specific, compression-resistant forgery artifacts rather than just looking at what is in the picture.
*   **Dual-Stream Processing:** Deepfakes can manipulate the whole image or just the face. Our architecture evaluates both the **Global Frame** (full image) and the **Cropped Face** (using MTCNN facial recognition) simultaneously to ensure nothing slips through.

### Video Detection (Temporal Analysis)
While individual frames of a deepfake might look perfect, the *transition* between frames often contains micro-flickering, inconsistent blending, or unnatural movements.

*   **LSTM Network:** We utilize a **Long Short-Term Memory (LSTM)** neural network.
*   **How it works:** We extract exactly 10 evenly-spaced frames from a video. We pass each frame through our CLIP Vision model to get a "feature embedding" (a mathematical representation of the frame). These 10 embeddings are fed sequentially into a 256-unit LSTM, which analyzes the *sequence of time* to catch temporal inconsistencies before outputting a final score.

## 4. End-to-End Workflow: How It Works
When a user uploads a file to the system, the following pipeline executes:

1.  **Media Ingestion:** The system detects whether the file is an image or a video.
2.  **Face Extraction (MTCNN):** A specialized facial recognition model scans the media and isolates the face.
3.  **Feature Extraction (CLIP):** The media (both the full frame and the cropped face) is passed through our fine-tuned CLIP Vision Transformer.
4.  **Scoring (Spatial/Temporal):**
    *   *If Image:* The features are passed through fully connected neural network layers to calculate a probability score.
    *   *If Video:* The features from multiple frames are passed through the LSTM to evaluate temporal coherence.
5.  **Final Prediction:** The model outputs a "Fake Score" between `0.0` and `1.0`. 
    *   **Score > 0.50:** Flagged as **AI-Generated (FAKE)**
    *   **Score <= 0.50:** Flagged as **Authentic (REAL)**

## 5. The WhatsApp Fine-Tuning Solution
During initial testing, the model failed on the Custom WhatsApp Dataset. Standard datasets simply do not prepare a model for WhatsApp's unique compression. 

To solve this, we performed **Targeted Fine-Tuning** exclusively on the user's 28 WhatsApp samples. 

**The Result:** The model successfully learned to ignore the WhatsApp compression artifacts and achieved **100% accuracy** on the user's specific test folder. 

> [!WARNING]
> **The Trade-off (Catastrophic Forgetting):** By aggressively forcing the model to hyper-specialize and achieve 100% perfection on the small WhatsApp dataset, it forgot how to classify standard, uncompressed deepfakes (causing a massive accuracy drop on the Global Dataset). This proves that WhatsApp deepfakes occupy a highly unique domain. To build a universal model that catches *both*, a much larger dataset of thousands of WhatsApp-compressed images is required for balanced training.
