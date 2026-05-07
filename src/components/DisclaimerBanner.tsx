import { useState } from 'react'
import './DisclaimerBanner.css'

export function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(
    () => sessionStorage.getItem('disclaimer_dismissed') === '1'
  )

  if (dismissed) return null

  return (
    <div className="disclaimer">
      <div className="disclaimer__inner">
        <svg className="disclaimer__icon" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
        </svg>
        <p className="disclaimer__text">
          <strong>Research Use Only.</strong> FetalyzeAI is a research-stage decision-support tool.
          It is NOT FDA-cleared, NOT clinically validated, and does NOT replace qualified clinicians.
          All predictions require expert clinical interpretation.
        </p>
        <button
          className="disclaimer__close"
          onClick={() => {
            sessionStorage.setItem('disclaimer_dismissed', '1')
            setDismissed(true)
          }}
          aria-label="Dismiss"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
          </svg>
        </button>
      </div>
    </div>
  )
}
