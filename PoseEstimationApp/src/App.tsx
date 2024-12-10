import React, { useState } from 'react';
import axios from 'axios';
import { Upload, BarChart2, Activity, Loader2, Download } from 'lucide-react';
import './index.css';

interface Results {
  annotatedVideo: string;
  poseDataCsv: string;
  thumbDistance: string;
  wristSpeed: string;
  wristX: string;
  wristY: string;
  wristZ: string;
}

const PoseEstimationApp = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Results | null>(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    setSelectedFile(file||null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file first');
      return;
    }

    const formData = new FormData();
    formData.append('video', selectedFile);

    setLoading(true);
    setError(null);
    setResults(null);
    setAnalysisResults(null);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults({
        annotatedVideo: response.data.annotated_video,
        poseDataCsv: response.data.pose_data_csv,
        thumbDistance: response.data.thumb_distance,
        wristSpeed:response.data.wrist_speed,
        wristX:response.data.wrist_x,
        wristY:response.data.wrist_y,
        wristZ:response.data.wrist_z,
      });
    } catch (err) {
      setError((err as any).response?.data?.error || 'An error occurred during upload');
    } finally {
      setLoading(false);
    }
  };

  const downloadAnnotatedVideo = async () => {
    if (!results?.annotatedVideo) {
      setError('No annotated video available');
      return;
    }

    try {
      const response = await axios({
        url: `http://localhost:5000${results.annotatedVideo}`,
        method: 'GET',
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'annotated_video.avi');
      document.body.appendChild(link);
      link.click();
      
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download annotated video');
      console.error(err);
    }
  };

  const downloadAnlysis = async () => {
    if (!results?.wristY) {
      setError('No annotated video available');
      return;
    }

    try {
      const response = await axios({
        url: `http://localhost:5000${results.wristY}`,
        method: 'GET',
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'wrist_y.png');
      document.body.appendChild(link);
      link.click();
      
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download analysis');
      console.error(err);
    }
  };

  const renderAnalysisResults = () => {
    if (!analysisResults) return null;

    return (
      <div className="mt-6 space-y-4">
        <div>
          <h2 className="text-xl font-semibold mb-4">Wrist Coordinate Plots</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <h3 className="text-lg font-semibold mb-4">X Coordinate</h3>
              {results && <img src={`http://localhost:5000${results.wristX}`} alt="Wrist X Coordinate" className="w-full rounded-lg" />}
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Y Coordinate</h3>
              {results && <img src={`http://localhost:5000${results.wristY}`} alt="Wrist Y Coordinate" className="w-full rounded-lg" />}
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Z Coordinate</h3>
              {results && <img src={`http://localhost:5000${results.wristZ}`} alt="Wrist Z Coordinate" className="w-full rounded-lg" />}
            </div>
          </div>
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-4">Thumb Distance</h2>
          {results && <img src={`http://localhost:5000${results.thumbDistance}`} alt="Thumb Distance" className="w-full rounded-lg" />}
        </div>
        <div>
          <h2 className="text-xl font-semibold mb-4">Thumb Distance</h2>
          {results && <img src={`http://localhost:5000${results.thumbDistance}`} alt="Thumb Distance" className="w-full rounded-lg" />}
        </div>
    </div>
    );
  };

  const renderResultActions = () => {
    if (!results) return null;

    return (
      <div className="mt-6 grid grid-cols-3 gap-4">
        <button
          onClick={downloadAnlysis}
          className="flex items-center justify-center p-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition"
        >
          <Activity className="mr-2" /> Analyze Data
        </button>
        <button
          onClick={downloadAnnotatedVideo}
          className="flex items-center justify-center p-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
        >
          <Download className="mr-2" /> Download Video
        </button>
        <button
          onClick={() => window.open(`http://localhost:5000${results.poseDataCsv}`, '_blank')}
          className="flex items-center justify-center p-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
        >
          <BarChart2 className="mr-2" /> Download CSV
        </button>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-2xl bg-white shadow-lg rounded-xl p-8">
        <h1 className="text-2xl font-bold mb-6 text-center text-gray-800">Pose Estimation Analysis</h1>
        <div className="flex items-center justify-center w-full">
          <label
            htmlFor="video-upload"
            className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="w-10 h-10 text-gray-500 mb-3" />
              <p className="mb-2 text-sm text-gray-500">
                {selectedFile ? `Selected: ${selectedFile.name}` : 'Click to upload or drag and drop'}
              </p>
              <p className="text-xs text-gray-400">MP4, AVI, MOV (Max 100MB)</p>
            </div>
            <input
              id="video-upload"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleFileChange}
            />
          </label>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-600 rounded-lg text-center">
            {error}
          </div>
        )}

        <div className="mt-6">
          <button
            onClick={handleUpload}
            disabled={!selectedFile || loading}
            className="w-full p-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition disabled:opacity-50 flex items-center justify-center"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 animate-spin" /> Processing...
              </>
            ) : (
              'Upload and Process Video'
            )}
          </button>
        </div>

        {renderResultActions()}
        {analysisResults && renderAnalysisResults()}
      </div>
    </div>
  );
};

export default PoseEstimationApp;