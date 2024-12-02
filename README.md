# Face Tracking with OSC

This Python application captures video from your webcam, tracks faces in real-time, and sends the tracking data via OSC (Open Sound Control).

## Features

- Real-time face detection using OpenCV
- Sends face position and size data via OSC
- Normalized coordinates (0-1) for easy integration
- Visual feedback with face detection rectangle

## OSC Message Details

The application sends the following OSC messages:

- `/face/x`: Normalized horizontal position of the face (0-1)
- `/face/y`: Normalized vertical position of the face (0-1)
- `/face/size`: Size of the detected face area

- `/mouth/opening`: Vertical distance between upper and lower lip (mouth opening)
- `/mouth/x`: Normalized horizontal position of the mouth (0-1)
- `/mouth/y`: Normalized vertical position of the mouth (0-1)

- `/eye/left/opening`: Vertical distance between upper and lower eyelid of the left eye (eye opening)
- `/eye/right/opening`: Vertical distance between upper and lower eyelid of the right eye (eye opening)
- `/eye/left/x`: Normalized horizontal position of the left eye (0-1)
- `/eye/left/y`: Normalized vertical position of the left eye (0-1)
- `/eye/right/x`: Normalized horizontal position of the right eye (0-1)
- `/eye/right/y`: Normalized vertical position of the right eye (0-1)

## Installation

Ensure you have the required dependencies installed as listed in `requirements.txt`. Additionally, download the `shape_predictor_68_face_landmarks.dat` file from [Dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it to the project directory.

## Usage

Run the script using:

```bash
python3 face_tracker.py
```

Select the appropriate camera index when prompted.

## Requirements

- Python 3.13
- OpenCV
- Dlib
- python-osc
- numpy

## Customization

You can modify the OSC IP address and port by changing the parameters when creating the FaceTracker:

```python
tracker = FaceTracker(osc_ip="192.168.1.100", osc_port=54321)
```
