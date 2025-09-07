A Python-based GUI application for enhancing grayscale chest X-ray images using various image processing techniques. Users can apply histogram equalization, gamma correction, and gamma + contrast stretching, and generate PDF reports with the results.

Features

Graphical User Interface (GUI) for easy interaction.

Supports multiple enhancement methods:

Histogram Equalization

Gamma Correction

Gamma + Contrast Stretching

Automatic generation of PDF reports with original and enhanced images along with their histograms.

Save enhanced images in a custom output directory.

Technologies Used

Python 3.x

OpenCV (cv2) for image processing

NumPy (numpy) for matrix operations

Matplotlib (matplotlib) for plotting histograms and generating PDF reports

Tkinter (tkinter) for GUI

Usage

Open the GUI by running:

python gui_script.py


Load a chest X-ray image.

Select an enhancement method.

Apply enhancement and view the result.

Save enhanced images and PDF reports.

Example

Original Image → Enhanced Image → Histogram comparison → PDF report

Audience

This tool is ideal for:

Medical imaging enthusiasts

Students learning image processing

Researchers in radiology or AI-based diagnostics
