# Handpose

![Project Banner](frontend/public/handpose.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Application](#running-the-application)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

**Handpose** is a web application designed for **2D and 3D pose estimation** from video files. Leveraging technologies like **MediaPipe** and **OpenCV**, it allows users to upload videos, perform pose and hand landmark detection, and visualize analytical data derived from processed frames. The application provides annotated videos, CSV data, and insightful plots to help users analyze movements effectively.

![Application Workflow](frontend/public/demo.png)

## Features

- **Video Upload**: Supports AVI, MP4, and MOV formats with a maximum size of 100MB.
- **Hand Selection**: Choose between left or right hand for targeted analysis.
- **Pose and Hand Detection**: Utilizes MediaPipe for accurate detection of pose and hand landmarks.
- **Annotated Videos**: Generates videos with drawn landmarks for visualization.
- **Data Export**:
    - **CSV Files**: Contains detailed pose and hand landmark data.
    - **Analytical Plots**: Includes wrist speed and thumb-index finger distance plots.
- **User-Friendly Interface**: React frontend with responsive design.
- **Dockerized Deployment**: Simplifies setup and ensures consistency.

## Tech Stack

- **Frontend**:
    - [React](https://reactjs.org/)
    - [Vite](https://vitejs.dev/)
    - [Tailwind CSS](https://tailwindcss.com/)
    - [TypeScript](https://www.typescriptlang.org/)

- **Backend**:
    - [Flask](https://flask.palletsprojects.com/)
    - [MediaPipe](https://mediapipe.dev/)
    - [OpenCV](https://opencv.org/)
    - [Pandas](https://pandas.pydata.org/)

- **DevOps**:
    - [Docker](https://www.docker.com/)
    - [Docker Compose](https://docs.docker.com/compose/)

## Architecture

The application follows a **client-server** architecture with a clear separation between the frontend and backend services. Docker Compose orchestrates these services, ensuring seamless communication and scalability.

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed on your machine.

### Installation

1. **Clone the Repository**

     ```bash
     git clone https://github.com/yourusername/Handpose.git
     cd Handpose
     ```

### Running the Application

1. **Build and Start Services**

     ```bash
     docker-compose up --build
     ```

     This command will:
     - Build Docker images for both frontend and backend.
     - Start containers and map ports:
         - Frontend: Accessible at http://localhost
         - Backend: Accessible at http://localhost:5000

2. **Access the Application**

     Open your web browser and navigate to http://localhost to access the Handpose interface.

3. **Stopping the Services**

     ```bash
     docker-compose down
     ```

## Usage

### Upload a Video

1. Click on the upload area to select a video file (AVI, MP4, MOV).
2. Ensure the file size does not exceed 100MB.

### Select Hand Preference

1. Choose either Left Hand or Right Hand for targeted analysis.

### Process the Video

1. Click on the "Upload and Process Video" button.
2. Wait for the processing to complete.

### Download Results

- **Annotated Video**: Download the video with drawn pose and hand landmarks.
- **Analysis Plots**: Download plots for wrist speed and thumb-index finger distance.
- **CSV Data**: Download the CSV file containing detailed landmark data.

![Demo Result](images/demo_result.png)

## API Endpoints

### POST /upload

- **Description**: Uploads a video file and processes it for pose and hand landmark detection.
- **Form Data**:
    - `video`: The video file to be processed.
    - `hand`: (Optional) left or right to specify the hand preference.
- **Response**:
    - `annotated_video`: URL to download the annotated video.
    - `pose_data_csv`: URL to download the CSV data.
    - `thumb_distance`: URL to download the thumb-index finger distance plot.
    - `wrist_speed`: URL to download the wrist speed plot.
    - `wrist_x`, `wrist_y`, `wrist_z`: URLs to download wrist coordinate plots.

### GET /outputs/<filename>

- **Description**: Serves processed files for download.
- **Parameters**:
    - `filename`: Name of the file to download.

## Project Structure

- `backend/`: Contains the Flask application and video processing scripts.
- `frontend/`: Contains the React application built with Vite and styled with Tailwind CSS.
- `docker-compose.yml`: Configuration to orchestrate frontend and backend services.
- `LICENSE`: MIT License for the project.
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! To get started:

1. **Fork the repository.**
2. **Create a new branch:**

     ```bash
     git checkout -b feature-branch
     ```

3. **Commit your changes:**

     ```bash
     git commit -m "Add new feature"
     ```

4. **Push to the branch:**

     ```bash
     git push origin feature-branch
     ```

5. **Open a Pull Request.**

Please ensure your code follows the project's coding standards.

## License

This project is licensed under the MIT License.

## Acknowledgements

- MediaPipe for providing versatile ML solutions.
- OpenCV for robust computer vision tools.
- React community for an excellent frontend framework.
- Docker for enabling seamless containerization.