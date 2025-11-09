import './App.css'
import VideoFeed from './components/VideoFeed'
import CameraControls from './components/CameraControls'
import AnomalyLog from './components/AnomalyLog'

function App() {
  return (
    <div className="min-h-screen starfield bg-space-gradient">
      {/* Main Container */}
      <div className="h-screen p-6 gap-6 grid grid-cols-1 lg:grid-cols-3 grid-rows-1">

        {/* Left Section - Video and Controls */}
  <div className="lg:col-span-2 flex flex-col gap-6">

          {/* Video Feed - Top */}
          <div className="flex-grow bg-black/50 rounded-3xl border border-cyan-500/20 p-6 shadow-glow">
            <VideoFeed />
          </div>

          {/* Camera Controls - Bottom */}
          <div className="bg-black/50 rounded-3xl border border-cyan-500/20 p-6 shadow-glow">
            <CameraControls />
          </div>
        </div>

        {/* Right Section - Anomaly Log */}
        <div className="lg:col-span-1 bg-black/50 rounded-3xl border border-cyan-500/20 p-6 shadow-glow">
          <AnomalyLog />
        </div>

      </div>
    </div>
  )
}

export default App
