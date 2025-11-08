import './App.css'
import VideoFeed from './components/VideoFeed'
import CameraControls from './components/CameraControls'
import AnomalyLog from './components/AnomalyLog'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Main Container */}
      <div className="h-screen p-6 gap-6 grid grid-cols-1 lg:grid-cols-3 grid-rows-1">

        {/* Left Section - Video and Controls */}
        <div className="lg:col-span-2 flex flex-col gap-6">

          {/* Video Feed - Top */}
          <div className="flex-grow bg-white rounded-3xl shadow-lg border border-blue-100 p-6">
            <VideoFeed />
          </div>

          {/* Controls and Data - Bottom */}
          <div className="h-48 bg-white rounded-3xl shadow-lg border border-blue-100 p-6">
            <CameraControls />
          </div>
        </div>

        {/* Right Section - Anomaly Log */}
        <div className="lg:col-span-1 bg-white rounded-3xl shadow-lg border border-blue-100 p-6">
          <AnomalyLog />
        </div>

      </div>
    </div>
  )
}

export default App
