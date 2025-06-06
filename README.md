# 📝 ClearText - Text Detection and Transcription Tool

An interactive OpenCV-based application for detecting text blocks in images and replacing them with manually transcribed text using mouse-based selection and background reconstruction.

## ✨ Features

### 🎯 Core Functionality
- **📷 Image Processing**: Automatic grayscale conversion and binary thresholding using Otsu's algorithm
- **🔍 Text Detection**: Connected components analysis to identify text regions
- **🖱️ Interactive Selection**: Mouse-based text block selection and management
- **✏️ Manual Transcription**: User-friendly text input for each detected block
- **🎨 Background Reconstruction**: Simple inpainting algorithm to remove original text
- **📄 Clean Output**: Final image with transcribed text rendered in appropriate fonts

### 🖱️ Mouse Interaction
- **Left Click**: Select text blocks
- **Right Click**: Deselect current block
- **Visual Feedback**: Color-coded blocks (red=unprocessed, green=transcribed, yellow=selected)
- **Real-time Preview**: Live display of transcription progress

### ⌨️ Keyboard Controls
- **`t`**: Transcribe selected text block
- **`d`**: Delete selected text block
- **`s`**: Save and continue to background reconstruction
- **`ESC`**: Exit application

## 🛠️ Technologies Used

- **📚 Framework**: OpenCV 4.x for computer vision operations
- **💻 Language**: C++ with STL containers
- **🖼️ Image Processing**: Custom implementations of morphological operations
- **🎨 GUI**: OpenCV's built-in window management and mouse callbacks
- **📊 Algorithms**: Otsu thresholding, connected components labeling, simple inpainting

## 🏗️ Algorithm Pipeline

### 1️⃣ **Image Preprocessing**
```cpp
// Grayscale conversion
grayscale = (R + G + B) / 3

// Automatic thresholding using Otsu's method
threshold = otsu_algorithm(histogram)
```

### 2️⃣ **Text Detection**
- Connected components analysis with 8-connectivity
- Size-based filtering for text regions
- Morphological dilation to connect nearby characters
- Horizontal dilation to merge words in same line

### 3️⃣ **Interactive Transcription**
- Mouse-based block selection system
- Manual text input for each region
- Progress tracking and validation

### 4️⃣ **Background Reconstruction**
- Mask creation from detected text regions
- Simple inpainting using neighbor averaging
- Multiple iterations for smooth reconstruction

### 5️⃣ **Text Rendering**
- Automatic font size calculation based on block dimensions
- Multi-line text wrapping
- Optimal positioning within original text boundaries

## 🚀 Installation & Usage

### 📋 Prerequisites
- **🔧 OpenCV 4.x**: Computer vision library
- **💻 Visual Studio**: Windows development environment
- **🖼️ Image Files**: JPG, PNG, BMP supported formats

### ▶️ Building the Project

1. **📥 Clone Repository**
   ```bash
   git clone <repository-url>
   cd cleartext-opencv
   ```

2. **⚙️ Configure OpenCV**
   - Install OpenCV 4.x
   - Set up include directories and library paths
   - Link required OpenCV modules (core, imgproc, imgcodecs, highgui)

3. **🏗️ Build**
   ```bash
   # Using Visual Studio
   # Open project file and build in Release mode
   ```

4. **▶️ Run Application**
   ```bash
   ./OpenCVApplication.exe
   ```

### 🎮 Usage Instructions

1. **📁 Load Image**: Select an image file through the file dialog
2. **👀 Review Detection**: Examine automatically detected text blocks
3. **🖱️ Select Blocks**: Click on text blocks to select them
4. **✏️ Transcribe**: Press `t` to input text for selected blocks
5. **🗑️ Delete**: Press `d` to remove unwanted blocks
6. **💾 Process**: Press `s` to proceed to background reconstruction
7. **📄 View Result**: Final image with transcribed text

## 🎨 Visual Workflow

### 🔄 Processing Pipeline
```
Original Image → Grayscale → Binary → Dilation → Text Detection
     ↓
Interactive Selection ← Mouse Callbacks ← Visual Display
     ↓
Manual Transcription → Background Inpainting → Text Rendering
     ↓
Final Clean Image
```

### 🎨 Color Coding System
- 🔴 **Red Blocks**: Unprocessed text regions
- 🟢 **Green Blocks**: Successfully transcribed
- 🟡 **Yellow Blocks**: Currently selected
- ⚪ **White Background**: Reconstructed areas

## 📊 Technical Details

### 🔍 Text Detection Criteria
- **📏 Area**: 10-5000 pixels
- **📐 Dimensions**: Width < 25% image width, Height < 10% image height
- **📏 Minimum Size**: 5x5 pixels
- **📄 Final Blocks**: Area > 200 pixels for paragraph detection

### 🎨 Font Rendering
- **📊 Automatic Sizing**: Font scale from 0.8 to 3.0
- **📝 Text Wrapping**: Intelligent word wrapping within block boundaries
- **📐 Positioning**: Optimal placement with padding considerations
- **🎯 Fallback**: Minimum font size for readability

### 🎨 Background Reconstruction
- **🎭 Mask Generation**: Binary mask from detected text pixels
- **🔄 Iterative Inpainting**: 10 iterations of neighbor averaging
- **🌈 Color Preservation**: RGB channel processing
- **📊 Edge Handling**: Boundary condition management

## 📁 Project Structure

```
src/
├── 📄 OpenCVApplication.cpp     # Main application file
├── 🔧 common.h                  # Common utilities and file dialogs
├── 📚 stdafx.h                  # Precompiled headers
└── 🎨 Image Processing Functions:
    ├── 🎯 grayscale_to_BW_autoClearText()     # Otsu thresholding
    ├── 🔍 labelConnectedComponentsClearText()  # Component analysis
    ├── 📏 dilateCustomClearText()              # Morphological operations
    ├── 🎨 simpleInpaintingClearText()         # Background reconstruction
    └── 🖱️ onMouseCallbackClearText()          # Mouse interaction
```

## 🎯 Key Algorithms

### 📊 **Otsu's Automatic Thresholding**
```cpp
// Iterative threshold calculation
while (|T_current - T_previous| >= error) {
    mean1 = average_intensity_below_threshold
    mean2 = average_intensity_above_threshold
    T_new = (mean1 + mean2) / 2
}
```

### 🔍 **Connected Components Analysis**
- 8-connectivity breadth-first search
- Bounding box calculation for each component
- Size-based filtering for text identification

### 🎨 **Simple Inpainting Algorithm**
```cpp
for each iteration:
    for each masked pixel:
        new_value = average(valid_neighbors)
```

## 🔄 Future Enhancements

- [ ] 🤖 **OCR Integration**: Automatic text recognition
- [ ] 🎨 **Advanced Inpainting**: Deep learning-based reconstruction
- [ ] 🌐 **Batch Processing**: Multiple image support
- [ ] 📊 **Better Font Matching**: Original font style preservation
- [ ] 🎯 **Region Splitting**: Manual block subdivision tools
- [ ] 💾 **Export Options**: Multiple output formats
- [ ] 🔄 **Undo/Redo**: Operation history management

## 🐛 Known Limitations

- **📝 Manual Transcription**: Requires user input for each text block
- **🎨 Simple Inpainting**: Basic algorithm may not handle complex backgrounds
- **🔤 Font Rendering**: Uses standard system fonts only
- **🖼️ Image Quality**: Works best with high-contrast text
- **💻 Platform**: Currently Windows-only implementation
