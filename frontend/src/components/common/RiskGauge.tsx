import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import './RiskGauge.css';

interface Props {
  score: number | null;
  band: string | null;
  color: string;
  processing?: boolean;
}

export function RiskGauge({ score, band, processing = false }: Props) {
  const scoreRef = useRef<HTMLSpanElement>(null);
  const arcRef = useRef<SVGCircleElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const prevScore = useRef<number>(0);

  const R = 45;
  const circumference = 2 * Math.PI * R;
  const spinArcLen = circumference * 0.28; // 28 % of ring visible while spinning

  const colorByBand = (b: string | null) => {
    switch (b) {
      case 'LOW': return '#10b981';
      case 'MEDIUM': return '#f59e0b';
      case 'HIGH': return '#f97316';
      case 'CRITICAL': return '#ef4444';
      default: return '#475569';
    }
  };

  const resolvedColor = colorByBand(band);

  /* ── Animate score + arc on result ── */
  useEffect(() => {
    if (score === null || processing) return;
    const numEl = scoreRef.current;
    const arcEl = arcRef.current;
    if (!numEl || !arcEl) return;

    const targetOffset = circumference - (score / 100) * circumference;

    const tl = gsap.timeline();

    const counter = { val: prevScore.current };
    tl.to(counter, {
      val: score,
      duration: 1.5,
      ease: 'power2.out',
      onUpdate: () => {
        numEl.textContent = Math.round(counter.val).toString();
      },
    }, 0);

    tl.to(arcEl, {
      strokeDashoffset: targetOffset,
      duration: 1.5,
      ease: 'power2.out',
    }, 0);

    tl.to(arcEl, {
      stroke: resolvedColor,
      duration: 0.8,
      ease: 'power1.inOut',
    }, 0.3);

    prevScore.current = score;
    return () => { tl.kill(); };
  }, [score, processing, circumference, resolvedColor]);

  /* ── Dots animation for center text while processing ── */
  const dotsRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!processing || !dotsRef.current) return;
    let i = 0;
    const iv = setInterval(() => {
      i = (i + 1) % 4;
      if (dotsRef.current) dotsRef.current.textContent = '.'.repeat(i);
    }, 400);
    return () => clearInterval(iv);
  }, [processing]);

  return (
    <div className="risk-gauge" ref={containerRef}>
      <div className={`risk-gauge__ring ${processing ? 'processing' : ''}`}>
        <svg viewBox="0 0 100 100" className="risk-gauge__svg">
          {/* Background track */}
          <circle
            cx="50" cy="50" r={R}
            fill="none"
            stroke="var(--border-primary)"
            strokeWidth="3.5"
            opacity="0.5"
          />

          {processing ? (
            /* ── Spinning arc (uses <g> to rotate so no transform conflict) ── */
            <g className="risk-gauge__spinner">
              <circle
                cx="50" cy="50" r={R}
                fill="none"
                stroke="url(#scanGradient)"
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray={`${spinArcLen} ${circumference - spinArcLen}`}
                strokeDashoffset={0}
              />
              {/* Glow trail (slightly wider, lower opacity) */}
              <circle
                cx="50" cy="50" r={R}
                fill="none"
                stroke="var(--primary)"
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${spinArcLen * 0.4} ${circumference - spinArcLen * 0.4}`}
                strokeDashoffset={0}
                opacity="0.12"
                className="risk-gauge__glow-trail"
              />
            </g>
          ) : (
            /* ── Static result arc ── */
            <circle
              ref={arcRef}
              cx="50" cy="50" r={R}
              fill="none"
              stroke={resolvedColor}
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference}
              transform="rotate(-90 50 50)"
            />
          )}

          {/* Gradient for the scanning arc */}
          <defs>
            <linearGradient id="scanGradient" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.15" />
              <stop offset="60%" stopColor="var(--primary)" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#4a82f0" stopOpacity="1" />
            </linearGradient>
          </defs>
        </svg>

        <div className="risk-gauge__center">
          {processing ? (
            <div className="risk-gauge__scanning">
              <span className="risk-gauge__scan-icon">
                <span className="material-icons" style={{ fontSize: 28 }}>radar</span>
              </span>
              <span className="risk-gauge__scan-label">SCANNING<span ref={dotsRef} className="risk-gauge__dots"></span></span>
            </div>
          ) : (
            <div className="risk-gauge__result">
              <span
                ref={scoreRef}
                className="risk-gauge__score"
                style={{ color: score !== null ? resolvedColor : 'var(--text-tertiary)' }}
              >
                {score !== null ? score : '—'}
              </span>
              <span className="risk-gauge__sub">/ 100</span>
            </div>
          )}
        </div>
      </div>

      <div
        className="risk-gauge__band"
        style={{ color: processing ? 'var(--primary)' : resolvedColor }}
      >
        {processing ? 'Analyzing...' : band || 'Awaiting Analysis'}
      </div>
    </div>
  );
}
