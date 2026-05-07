import './Header.css'

export function Header() {
  return (
    <header className="header">
      <div className="header__inner">
        <div className="header__brand">
          <div className="header__logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect width="28" height="28" rx="8" fill="#1d4ed8" />
              <path d="M7 14c0-3.866 3.134-7 7-7s7 3.134 7 7" stroke="#bfdbfe" strokeWidth="2" strokeLinecap="round" fill="none"/>
              <path d="M5 14h2.5l2-4 3 8 2.5-6 1.5 2H23" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
            </svg>
          </div>
          <div>
            <h1 className="header__title">FetalyzeAI</h1>
            <p className="header__subtitle">CTG Second-Reader &middot; v4</p>
          </div>
        </div>
        <div className="header__badge">
          <span className="header__badge-dot" />
          Research Stage
        </div>
      </div>
    </header>
  )
}
