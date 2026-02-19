import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import './WelcomeScreen.css';

export function WelcomeScreen() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const tl = gsap.timeline();
    tl.fromTo(el.querySelector('.welcome__logo'),
      { scale: 0.8, opacity: 0 },
      { scale: 1, opacity: 1, duration: 0.6, ease: 'back.out(1.4)' }
    );
    tl.fromTo(el.querySelector('.welcome__tagline'),
      { y: 10, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.5, ease: 'power2.out' },
      '-=0.3'
    );
    tl.fromTo(el.querySelectorAll('.welcome__step'),
      { y: 20, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.4, ease: 'power2.out', stagger: 0.12 },
      '-=0.2'
    );

    return () => { tl.kill(); };
  }, []);

  return (
    <div className="welcome" ref={containerRef}>
      <div className="welcome__logo">HATAD</div>
      <div className="welcome__tagline">
        AI-Powered Land Due Diligence for Tamil Nadu
      </div>
      <div className="welcome__steps">
        <div className="welcome__step">
          <div className="welcome__step-icon">
            <span className="material-icons">upload_file</span>
          </div>
          <div className="welcome__step-num">01</div>
          <div className="welcome__step-title">Upload Documents</div>
          <div className="welcome__step-desc">EC, Patta, Sale Deed, FMB & more</div>
        </div>
        <div className="welcome__step-arrow">
          <span className="material-icons">arrow_forward</span>
        </div>
        <div className="welcome__step">
          <div className="welcome__step-icon">
            <span className="material-icons">psychology</span>
          </div>
          <div className="welcome__step-num">02</div>
          <div className="welcome__step-title">AI Analysis</div>
          <div className="welcome__step-desc">Multi-pass verification & fraud detection</div>
        </div>
        <div className="welcome__step-arrow">
          <span className="material-icons">arrow_forward</span>
        </div>
        <div className="welcome__step">
          <div className="welcome__step-icon">
            <span className="material-icons">assessment</span>
          </div>
          <div className="welcome__step-num">03</div>
          <div className="welcome__step-title">Get Report</div>
          <div className="welcome__step-desc">Risk score, findings & recommendations</div>
        </div>
      </div>
    </div>
  );
}
