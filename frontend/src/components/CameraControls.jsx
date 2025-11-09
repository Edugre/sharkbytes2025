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
    sendCommand('center')
    setPanAngle(90)
    setTiltAngle(90)
  }

  const handlePan = (direction) => {
    const step = 5
    const newAngle = direction === 'left'
      ? Math.max(10, panAngle - step)
      : Math.min(170, panAngle + step)
    setPanAngle(newAngle)
    // Swap left/right to fix inversion
    sendCommand(direction === 'left' ? 'pan_right' : 'pan_left')
  }

  const handleTilt = (direction) => {
    const step = 5
    const newAngle = direction === 'up'
      ? Math.max(20, tiltAngle - step)
      : Math.min(150, tiltAngle + step)
    setTiltAngle(newAngle)
    // Swap up/down to fix inversion
    sendCommand(direction === 'up' ? 'tilt_down' : 'tilt_up')
  }

  return (
    <div className="w-full h-full flex items-center justify-between text-slate-100">
      <h3 className="text-sm font-semibold neon-text tracking-wide">Camera Controls</h3>
      
      {/* Inline Control Buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => handlePan('left')}
          className="w-10 h-10 rounded-md bg-black/60 border border-cyan-500/30 text-cyan-200 hover:text-cyan-100 hover:border-cyan-400 hover:bg-cyan-500/10 active:scale-95 flex items-center justify-center font-bold text-lg transition-all"
        >
          ←
        </button>
        <button
          onClick={() => handleTilt('up')}
          className="w-10 h-10 rounded-md bg-black/60 border border-cyan-500/30 text-cyan-200 hover:text-cyan-100 hover:border-cyan-400 hover:bg-cyan-500/10 active:scale-95 flex items-center justify-center font-bold text-lg transition-all"
        >
          ↑
        </button>
        <button
          onClick={handleCenter}
          className="px-3 h-10 rounded-md bg-gradient-to-br from-cyan-500 to-blue-600 text-slate-900 font-semibold text-xs hover:brightness-110 active:scale-95 flex items-center justify-center transition-all"
        >
          center
        </button>
        <button
          onClick={() => handleTilt('down')}
          className="w-10 h-10 rounded-md bg-black/60 border border-cyan-500/30 text-cyan-200 hover:text-cyan-100 hover:border-cyan-400 hover:bg-cyan-500/10 active:scale-95 flex items-center justify-center font-bold text-lg transition-all"
        >
          ↓
        </button>
        <button
          onClick={() => handlePan('right')}
          className="w-10 h-10 rounded-md bg-black/60 border border-cyan-500/30 text-cyan-200 hover:text-cyan-100 hover:border-cyan-400 hover:bg-cyan-500/10 active:scale-95 flex items-center justify-center font-bold text-lg transition-all"
        >
          →
        </button>
      </div>

      {/* Manual/Auto Toggle */}
      <button
        onClick={handleToggleLock}
        className={`px-3 py-1.5 rounded-md font-semibold text-xs tracking-wide transition-all ${
          isLocked
            ? 'bg-cyan-500 text-slate-900 hover:brightness-110'
            : 'bg-slate-800 border border-cyan-500/30 text-cyan-200 hover:border-cyan-400'
        }`}
      >
        {isLocked ? 'AUTO' : 'MANUAL'}
      </button>
    </div>
  )
}

export default CameraControls
