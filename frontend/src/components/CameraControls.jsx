import { useState } from 'react'

function CameraControls() {
  const [isLocked, setIsLocked] = useState(false)
  const [panAngle, setPanAngle] = useState(90)
  const [tiltAngle, setTiltAngle] = useState(90)

  // API endpoint - adjust to match your backend
  const API_BASE = 'http://localhost:5000'

  // Send command to backend
  const sendCommand = async (command) => {
    try {
      const response = await fetch(`${API_BASE}/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command })
      })
      if (!response.ok) throw new Error('Command failed')
    } catch (error) {
      console.error('Control error:', error)
    }
  }

  const handleToggleLock = () => {
    sendCommand('toggle_lock')
    setIsLocked(!isLocked)
  }

  const handleCenter = () => {
    if (!isLocked) {
      sendCommand('center')
      setPanAngle(90)
      setTiltAngle(90)
    }
  }

  const handlePan = (direction) => {
    const step = 5
    const newAngle = direction === 'left'
      ? Math.max(10, panAngle - step)
      : Math.min(170, panAngle + step)
    setPanAngle(newAngle)
    sendCommand(`pan_${direction}`)
  }

  const handleTilt = (direction) => {
    const step = 5
    const newAngle = direction === 'up'
      ? Math.max(20, tiltAngle - step)
      : Math.min(150, tiltAngle + step)
    setTiltAngle(newAngle)
    sendCommand(`tilt_${direction}`)
  }

  return (
    <div className="w-full h-full flex flex-col justify-center">
      <h3 className="text-sm font-semibold text-slate-600 mb-4">Camera Controls</h3>

      {/* All Controls in One Row */}
      <div className="flex items-center gap-3">
        {/* Pan Left */}
        <button
          onClick={() => handlePan('left')}
          disabled={isLocked}
          className={`w-14 h-14 rounded-xl ${
            isLocked
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700'
          } transition-colors duration-150 flex items-center justify-center font-bold text-xl shadow-md`}
        >
          ←
        </button>

        {/* Tilt Up */}
        <button
          onClick={() => handleTilt('up')}
          disabled={isLocked}
          className={`w-14 h-14 rounded-xl ${
            isLocked
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700'
          } transition-colors duration-150 flex items-center justify-center font-bold text-xl shadow-md`}
        >
          ↑
        </button>

        {/* Tilt Down */}
        <button
          onClick={() => handleTilt('down')}
          disabled={isLocked}
          className={`w-14 h-14 rounded-xl ${
            isLocked
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700'
          } transition-colors duration-150 flex items-center justify-center font-bold text-xl shadow-md`}
        >
          ↓
        </button>

        {/* Pan Right */}
        <button
          onClick={() => handlePan('right')}
          disabled={isLocked}
          className={`w-14 h-14 rounded-xl ${
            isLocked
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600 active:bg-blue-700'
          } transition-colors duration-150 flex items-center justify-center font-bold text-xl shadow-md`}
        >
          →
        </button>

        {/* Center Button */}
        <button
          onClick={handleCenter}
          disabled={isLocked}
          className={`w-14 h-14 rounded-xl ${
            isLocked
              ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
              : 'bg-slate-600 text-white hover:bg-slate-700 active:bg-slate-800'
          } transition-colors duration-150 flex items-center justify-center font-bold text-2xl shadow-md`}
        >
          ⊙
        </button>

        {/* Divider */}
        <div className="h-12 w-px bg-slate-300 mx-2"></div>

        {/* Lock/Unlock Toggle */}
        <button
          onClick={handleToggleLock}
          className={`px-6 h-14 rounded-xl font-semibold text-sm shadow-lg transition-all duration-200 ${
            isLocked
              ? 'bg-green-500 text-white hover:bg-green-600 active:bg-green-700'
              : 'bg-slate-300 text-slate-700 hover:bg-slate-400 active:bg-slate-500'
          }`}
        >
          {isLocked ? '🔒 Locked' : '🔓 Unlocked'}
        </button>

        {/* Status Text */}
        <div className="ml-2">
          <p className="text-xs text-slate-500">
            {isLocked ? 'Auto-tracking' : 'Manual mode'}
          </p>
        </div>
      </div>
    </div>
  )
}

export default CameraControls
