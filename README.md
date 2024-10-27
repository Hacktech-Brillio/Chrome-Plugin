# 🔍 Brillio AI Fake Review Detector - Chrome Extension

The **AI Fake Review Detector** Chrome extension is designed to identify computer-generated reviews on e-commerce and review websites, helping users make informed purchasing decisions with reliable and authentic information.

## 📋 Purpose

This extension detects potentially fake reviews on any webpage, offering users clear indicators for "Fake Review" or "Original Review" to support trusted decision-making.

## ✨ Features

- **Fake Review Detection**: Automatically detects and highlights suspicious reviews.
- **Visual Indicators**: Labels each review as "Fake" or "Original" for easy identification.

## 🚀 How It Works

1. **Open a Review Page**: Navigate to a webpage containing reviews.
2. **Activate the Extension**: Click on the extension icon and select **"Analyze Reviews"**.
3. **Review Results**: The extension processes each review, displaying a label to mark it as "Fake" or "Original."

## 🛠️ Technical Architecture

The extension is composed of four main components:

1. **Popup**: The user interface for triggering review analysis.
2. **Background Script**: Manages overall extension behavior and injects content scripts.
3. **Content Script**: Injects AI-powered logic to analyze reviews directly on the webpage.
4. **AI Model**: Uses a pre-trained ONNX model to classify reviews as genuine or fake.

### Technologies Used

- **JavaScript**: Core extension functionality.
- **Python & ONNX**: For model training and export.
- **Chrome Extension APIs**: Uses ActiveTab, scripting, and storage APIs.

### AI Model

- **Model Training**: The model was trained on labeled data, classifying reviews as either **CG** (fake) or **OR** (original).
- **Preprocessing Steps**: Text cleaning, tokenization, stop word removal, and word embeddings.
- **Architecture**: Bidirectional LSTM layer built in PyTorch and exported to ONNX for browser compatibility.

### Content Script

- **Function**: Parses the webpage to identify and analyze reviews.
- **Visualization**: Appends a "Fake Review" or "Original Review" label next to each review based on the AI’s classification.

## 📝 User Guide

1. **Open a Review Page**: Navigate to a website with product or service reviews.
2. **Analyze Reviews**: Click the extension icon, then click the "Analyze Reviews" button.
3. **View Results**: Reviews are labeled on the page as either "Fake Review" or "Original Review."

## 🚧 Challenges and Considerations

- **Content Script Injection**: Ensuring efficient loading and execution for smooth analysis.
- **ONNX Model Integration**: Achieving optimized inference in the browser to maintain performance.

## 🌐 Future Improvements

- **Confidence Scores**: Indicate the model’s certainty for each review.
- **Language Support**: Expand analysis capabilities for non-English reviews.
- **UI Enhancements**: Add tooltips to flagged reviews for context and a dashboard for tracking analyzed reviews and overall statistics.

## 📜 Conclusion

The AI Fake Review Detector Chrome extension brings transparency to online reviews, enabling users to make more informed purchasing decisions by identifying computer-generated or misleading reviews. With planned enhancements, the extension will continue to improve accuracy and user experience, adding more value for users in their online shopping journey.
