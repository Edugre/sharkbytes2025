import { useState, useEffect } from 'react'

function VideoFeed() {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState(null)

  // Placeholder for video stream URL - you'll configure this to point to your OpenCV stream
  const streamUrl = 'http://localhost:5000/video_feed' // Adjust this to match your backend

  useEffect(() => {
    // Simulate connection check
    const checkConnection = () => {
      // This is a placeholder - you can add actual connection logic later
      setIsConnected(true)
    }

    checkConnection()
  }, [])

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-slate-700">Live Camera Feed</h2>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
          <span className="text-sm text-slate-500">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Video Container */}
      <div className="flex-grow bg-slate-900 rounded-2xl overflow-hidden relative">
        {error ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center">
              <p className="text-red-400 mb-2">Connection Error</p>
              <p className="text-slate-400 text-sm">{error}</p>
            </div>
          </div>
        ) : (
          <img
            src={streamUrl}
            alt="Camera Feed"
            className="w-full h-full object-contain"
            onError={() => setError('Unable to connect to camera stream')}
            onLoad={() => setError(null)}
          />
        )}

        {/* Overlay info - optional */}
        <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-lg">
          <p className="text-white text-xs font-mono">OpenCV Stream</p>
        </div>
      </div>
    </div>
  )
}

export default VideoFeed
