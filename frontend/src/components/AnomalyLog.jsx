import { useState, useEffect } from 'react'

function AnomalyLog() {
  const [anomalies, setAnomalies] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedAnomaly, setSelectedAnomaly] = useState(null)
  const [deleting, setDeleting] = useState(false)

  // API endpoint - adjust to match your backend
  const API_BASE = 'http://localhost:5000'

  // Fetch anomalies from the database
  useEffect(() => {
    const fetchAnomalies = async () => {
      try {
        const response = await fetch(`${API_BASE}/events?limit=50`)
        if (!response.ok) throw new Error('Failed to fetch events')
        const data = await response.json()

        // Transform backend data to match frontend expectations
        const transformedData = data.map(event => ({
          id: event.id,
          timestamp: event.timestamp,
          description: event.description,
          severity: event.severity, // info/warning/critical from backend
          imageUrl: event.image_url
        }))

        setAnomalies(transformedData)
        setLoading(false)
      } catch (error) {
        console.error('Error fetching anomalies:', error)
        setLoading(false)
        setAnomalies([]) // Empty array when backend is unavailable
      }
    }

    fetchAnomalies()

    // Poll for new anomalies every 5 seconds
    const interval = setInterval(fetchAnomalies, 5000)
    return () => clearInterval(interval)
  }, [])

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    })
  }

  const formatDate = (timestamp) => {
    const date = new Date(timestamp)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    })
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-700 border-red-200'
      case 'warning':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200'
      case 'info':
        return 'bg-blue-100 text-blue-700 border-blue-200'
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200'
    }
  }

  const handleDeleteEvent = async (eventId) => {
    if (!confirm('Are you sure you want to delete this event? This action cannot be undone.')) {
      return
    }

    setDeleting(true)
    try {
      const response = await fetch(`${API_BASE}/events/${eventId}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        throw new Error('Failed to delete event')
      }

      // Remove the deleted event from the list
      setAnomalies(anomalies.filter(a => a.id !== eventId))

      // Close the modal
      setSelectedAnomaly(null)

      console.log(`Event ${eventId} deleted successfully`)
    } catch (error) {
      console.error('Error deleting event:', error)
      alert('Failed to delete event. Please try again.')
    } finally {
      setDeleting(false)
    }
  }

  return (
  <div className="w-full h-full flex flex-col text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
  <h2 className="text-xl font-semibold neon-text">Anomaly Log</h2>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></div>
          <span className="text-xs text-slate-500">Live</span>
        </div>
      </div>

      {/* Anomaly List */}
  <div className="flex-grow overflow-y-auto space-y-3 pr-2">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-slate-300 text-sm">Loading anomalies...</div>
          </div>
        ) : anomalies.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-slate-300 text-sm mb-2">No anomalies detected</p>
              <p className="text-slate-400 text-xs">System monitoring active</p>
            </div>
          </div>
        ) : (
          anomalies.map((anomaly) => (
            <div
              key={anomaly.id}
              onClick={() => setSelectedAnomaly(anomaly)}
              className="bg-black/40 rounded-xl p-4 border border-cyan-500/10 hover:border-cyan-400/30 hover:shadow-glow transition-all duration-200 cursor-pointer"
            >
              {/* Timestamp and Severity */}
              <div className="flex items-start justify-between mb-2">
                <div className="text-xs text-slate-300">
                  <span className="font-semibold">{formatDate(anomaly.timestamp)}</span>
                  <span className="mx-1">•</span>
                  <span>{formatTimestamp(anomaly.timestamp)}</span>
                </div>
                <span
                  className={`px-2 py-0.5 rounded-md text-xs font-medium border ${getSeverityColor(
                    anomaly.severity
                  )}`}
                >
                  {anomaly.severity.toUpperCase()}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-slate-200 leading-relaxed mb-3">
                {anomaly.description}
              </p>

              {/* Image Preview (if available) */}
              {anomaly.imageUrl && (
                <div className="rounded-lg overflow-hidden bg-black/60 border border-cyan-500/10">
                  <img
                    src={anomaly.imageUrl}
                    alt="Anomaly capture"
                    className="w-full h-32 object-cover"
                  />
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Footer Stats */}
      <div className="mt-4 pt-4 border-t border-cyan-500/10">
        <div className="flex items-center justify-between text-xs text-slate-300">
          <span>Total Events: {anomalies.length}</span>
          <span>Last 24h</span>
        </div>
      </div>

      {/* Modal */}
      {selectedAnomaly && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedAnomaly(null)}
        >
          <div
            className="bg-black/70 border border-cyan-500/20 rounded-2xl max-w-4xl max-h-[90vh] overflow-auto shadow-glow"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="sticky top-0 bg-black/60 border-b border-cyan-500/10 p-6 flex items-start justify-between">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-2xl font-semibold neon-text">Event Details</h3>
                  <span
                    className={`px-3 py-1 rounded-lg text-sm font-medium border ${getSeverityColor(
                      selectedAnomaly.severity
                    )}`}
                  >
                    {selectedAnomaly.severity.toUpperCase()}
                  </span>
                </div>
                <div className="text-sm text-slate-300">
                  <span className="font-semibold">{formatDate(selectedAnomaly.timestamp)}</span>
                  <span className="mx-2">•</span>
                  <span>{formatTimestamp(selectedAnomaly.timestamp)}</span>
                </div>
              </div>
              <button
                onClick={() => setSelectedAnomaly(null)}
                className="text-slate-300 hover:text-white transition-colors p-2 hover:bg-white/10 rounded-lg"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6">
              {/* Description */}
              <div className="mb-6">
                <h4 className="text-sm font-semibold text-slate-200 mb-2">Description</h4>
                <p className="text-slate-200 leading-relaxed">
                  {selectedAnomaly.description}
                </p>
              </div>

              {/* Full Image */}
              {selectedAnomaly.imageUrl && (
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-slate-200 mb-2">Captured Image</h4>
                  <div className="rounded-xl overflow-hidden bg-black/60 border border-cyan-500/10">
                    <img
                      src={selectedAnomaly.imageUrl}
                      alt="Anomaly capture full size"
                      className="w-full h-auto object-contain"
                    />
                  </div>
                </div>
              )}

              {/* Delete Button */}
              <div className="flex justify-end pt-4 border-t border-cyan-500/10">
                <button
                  onClick={() => handleDeleteEvent(selectedAnomaly.id)}
                  disabled={deleting}
                  className="px-4 py-2 bg-red-600/80 hover:bg-red-600 text-white rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed border border-red-500/30 hover:shadow-lg hover:shadow-red-500/20"
                >
                  {deleting ? 'Deleting...' : 'Delete Event'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AnomalyLog
