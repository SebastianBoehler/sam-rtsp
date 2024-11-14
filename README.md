# SAM-RTSP 🎥✨

Real-time human segmentation for RTSP camera streams using Meta's Segment Anything Model (SAM) and YOLO. This project combines the power of YOLO's fast object detection with SAM's precise segmentation capabilities to create high-quality human segmentation masks from live camera feeds.

![SAM-RTSP Demo](demo.gif) *(Add your demo gif here)*

## 🚀 Features

- 🎯 Real-time human detection using YOLOv8n
- 🎭 Precise segmentation masks using Meta's SAM
- 📹 RTSP camera stream support with auto-reconnection
- 🖥️ High-resolution display with optimized processing
- 🎚️ Configurable confidence threshold (default: 70%)
- 🔒 Secure credential management

## 🛠️ Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- RTSP camera stream

## ⚡️ Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sam-rtsp.git
cd sam-rtsp
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the SAM checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

5. Configure your RTSP stream:
```bash
cp .env.template .env
# Edit .env with your camera credentials
```

6. Run the script:
```bash
python rtsp_sam_human_segmentation.py
```

## ⚙️ Configuration

1. Copy `.env.template` to `.env`
2. Edit `.env` with your RTSP camera details:
```env
RTSP_USERNAME=your_username
RTSP_PASSWORD=your_password
RTSP_IP=your_camera_ip
RTSP_PORT=554
RTSP_CHANNEL=101
```

## 🎮 Usage

The script will:
1. Connect to your RTSP stream
2. Detect humans using YOLO (>70% confidence)
3. Generate precise segmentation masks using SAM
4. Display the results in real-time

### Parameters

- `process_every_n_frames`: Control processing frequency (default: 30)
- `process_width`: Width for processing (default: 640)
- `process_height`: Height for processing (default: 480)
- `confidence_threshold`: YOLO detection threshold (default: 0.7)

## 🚀 Performance

- Efficient processing using frame downscaling
- Smart frame skipping for reduced CPU/GPU load
- Confidence thresholding to minimize false positives
- Lightweight YOLOv8n model for fast detection

## 🔒 Security

- Credentials stored in `.env` (not committed)
- Template provided for safe sharing
- Sensitive files excluded via `.gitignore`

## 🔧 Troubleshooting

1. Stream Connection:
   - Verify RTSP URL format
   - Check network connectivity
   - Confirm credentials in `.env`

2. Performance:
   - Increase frame skip rate
   - Lower processing resolution
   - Adjust confidence threshold

## 📜 License

MIT License - feel free to use in your own projects!

## 🙏 Acknowledgments

- [Segment Anything Model (SAM)](https://segment-anything.com/) by Meta AI
- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- [OpenCV](https://opencv.org/) for image processing

## 📸 Examples

*(Add screenshots/gifs of your segmentation results here)*

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.
