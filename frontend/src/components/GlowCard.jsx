import React from 'react';

// Reusable panel wrapper
// Props:
//  - title?: optional heading string
//  - actions?: optional React node for top-right actions
//  - className?: extra classes appended to outer container
//  - children: content
export default function GlowCard({ title, actions, className = '', children }) {
  return (
    <div className={`bg-black/50 border border-cyan-500/15 rounded-2xl shadow-glow hover:border-cyan-400/30 transition-colors ${className}`}>
      {(title || actions) && (
        <div className="flex items-start justify-between p-4 pb-0">
          {title && <h3 className="font-semibold tracking-wide neon-text text-lg">{title}</h3>}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
