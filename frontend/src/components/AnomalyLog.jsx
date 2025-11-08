import { useState, useEffect } from 'react'

function AnomalyLog() {
  const [anomalies, setAnomalies] = useState([])
  const [loading, setLoading] = useState(true)

  // API endpoint - adjust to match your backend
  const API_BASE = 'http://localhost:5000'

  // Fetch anomalies from the database
  useEffect(() => {
    const fetchAnomalies = async () => {
      try {
        const response = await fetch(`${API_BASE}/anomalies`)
        if (!response.ok) throw new Error('Failed to fetch anomalies')
        const data = await response.json()
        setAnomalies(data)
        setLoading(false)
      } catch (error) {
        console.error('Error fetching anomalies:', error)
        setLoading(false)
        // Use mock data for testing when backend is not available
        setAnomalies([
          {
            id: 1,
            timestamp: '2025-11-08T10:30:15',
            description: 'Person detected entering restricted area at north entrance',
            severity: 'high',
            imageUrl: null
          },
          {
            id: 2,
            timestamp: '2025-11-08T09:15:42',
            description: 'Unusual movement pattern detected near storage room',
            severity: 'medium',
            imageUrl: null
          },
          {
            id: 3,
            timestamp: '2025-11-08T08:45:30',
            description: 'Multiple persons detected in normally empty corridor',
            severity: 'low',
            imageUrl: null
          }
        ])
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
      case 'high':
        return 'bg-red-100 text-red-700 border-red-200'
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 border-yellow-200'
      case 'low':
        return 'bg-blue-100 text-blue-700 border-blue-200'
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200'
    }
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-slate-700">Anomaly Log</h2>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></div>
          <span className="text-xs text-slate-500">Live</span>
        </div>
      </div>

      {/* Anomaly List */}
      <div className="flex-grow overflow-y-auto space-y-3 pr-2">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-slate-400 text-sm">Loading anomalies...</div>
          </div>
        ) : anomalies.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <p className="text-slate-400 text-sm mb-2">No anomalies detected</p>
              <p className="text-slate-300 text-xs">System monitoring active</p>
            </div>
          </div>
        ) : (
          anomalies.map((anomaly) => (
            <div
              key={anomaly.id}
              className="bg-slate-50 rounded-xl p-4 border border-slate-200 hover:shadow-md transition-shadow duration-200"
            >
              {/* Timestamp and Severity */}
              <div className="flex items-start justify-between mb-2">
                <div className="text-xs text-slate-500">
                  <span className="font-semibold">{formatDate(anomaly.timestamp)}</span>
                  <span className="mx-1">•</span>
                  <span>{formatTimestamp(anomaly.timestamp)}</span>
                </div>
                <span
                  className={`px-2 py-0.5 rounded-md text-xs font-medium border ${getSeverityColor(
                    anomaly.severity
                  )}`}
                >
                  {anomaly.severity}
                </span>
              </div>

              {/* Description */}
              <p className="text-sm text-slate-700 leading-relaxed mb-3">
                {anomaly.description}
              </p>

              {/* Image Preview (if available) */}
              {anomaly.imageUrl && (
                <div className="rounded-lg overflow-hidden bg-slate-900">
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
      <div className="mt-4 pt-4 border-t border-slate-200">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>Total Events: {anomalies.length}</span>
          <span>Last 24h</span>
        </div>
      </div>
    </div>
  )
}

export default AnomalyLog
