import { useState, useEffect } from 'react'

function VideoFeed() {
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState(null)
  const [systemStatus, setSystemStatus] = useState({
    sentry_running: false,
    sentry_available: false,
  })
  const [stats, setStats] = useState(null)
  const [actionLoading, setActionLoading] = useState(false)

  // Placeholder for video stream URL - you'll configure this to point to your OpenCV stream
  const streamUrl = 'http://localhost:5000/video_feed' // Adjust this to match your backend
  const API_URL = 'http://localhost:5000'

  useEffect(() => {
    // Simulate connection check
    const checkConnection = () => {
      // This is a placeholder - you can add actual connection logic later
      setIsConnected(true)
    }

    checkConnection()
    
    // Fetch system status
    fetchStatus()
    const interval = setInterval(fetchStatus, 2000) // Update every 2 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/system/status`)
      const data = await response.json()
      setSystemStatus({
        sentry_running: data.sentry_running,
        sentry_available: data.sentry_available,
      })
      setStats(data.stats)
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    }
  }

  const handleToggle = async () => {
    setActionLoading(true)
    try {
      const endpoint = systemStatus.sentry_running ? '/system/stop' : '/system/start'
      const response = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
      })
      const data = await response.json()
      console.log('Toggle response:', data)
      await fetchStatus()
    } catch (error) {
      console.error('Failed to toggle system:', error)
    } finally {
      setActionLoading(false)
    }
  }

  return (
    <div className="w-full h-full flex flex-col text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold neon-text">Live Camera Feed</h2>
        
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-cyan-400' : 'bg-red-400'} animate-pulse`}></div>
            <span className="text-sm text-slate-300">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          {/* Start/Stop Toggle Button */}
          <button
            onClick={handleToggle}
            disabled={actionLoading || !systemStatus.sentry_available}
            className={`p-2 rounded-lg transition-all focus-neon shadow-glow ${
              !systemStatus.sentry_available
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : systemStatus.sentry_running
                  ? 'bg-red-600 text-white hover:bg-red-500 active:scale-95'
                  : 'bg-[var(--accent)] text-slate-900 hover:brightness-110 active:scale-95'
            }`}
            title={systemStatus.sentry_running ? 'Stop Sentry' : 'Start Sentry'}
          >
            {actionLoading ? (
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : systemStatus.sentry_running ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Video Container */}
      <div className="flex-grow rounded-2xl overflow-hidden relative border border-cyan-500/20 bg-black/40">
        {error ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center">
              <p className="text-red-400 mb-2">Connection Error</p>
              <p className="text-slate-300 text-sm">{error}</p>
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

        {/* Overlay info */}
        <div className="absolute top-4 left-4 flex gap-3">
          {/* LIVE Indicator */}
          <div className="bg-black/70 px-3 py-1.5 rounded-md border border-cyan-500/40">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"></div>
              <p className="text-cyan-100 text-xs font-semibold tracking-wider">LIVE</p>
            </div>
          </div>

          {/* Stats */}
          {systemStatus.sentry_running && stats && (
            <div className="bg-black/70 px-3 py-1.5 rounded-md border border-cyan-500/40">
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="text-cyan-400 font-semibold">STATUS</span>
                  <span className="text-slate-200">{stats.tracking_status}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-cyan-400 font-semibold">FPS</span>
                  <span className="text-slate-200">{Math.round(stats.fps)}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-cyan-400 font-semibold">TRACKS</span>
                  <span className="text-slate-200">{stats.people_count}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default VideoFeed
