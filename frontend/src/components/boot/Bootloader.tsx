/* ═══════════════════════════════════════════════════
   HATAD — Bootloader
   Clean branded splash. System checks run silently;
   errors surface only when they matter.
   ═══════════════════════════════════════════════════ */

import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import gsap from 'gsap';
import { HatadLogo } from './HatadLogo';
import { useBootCheck, type BootPhase } from '../../hooks/useBootCheck';
import './Bootloader.css';

/* Friendly names shown only on failure */
const FRIENDLY: Record<string, string> = {
  backend: 'Backend API',
  filesystem: 'File System',
  ollama: 'Language Runtime',
  reasoning_model: 'Reasoning Engine',
  embedding_model: 'Embeddings',
  sarvam: 'Tamil OCR',
};

const DISPLAY_ORDER = [
  'backend', 'filesystem', 'ollama',
  'reasoning_model', 'embedding_model', 'sarvam',
];

/* ── Loading hint lines (cycle through while waiting) ── */
const HINTS = [
  'Preparing your workspace',
  'Loading analysis models',
  'Connecting services',
];

export function Bootloader({ onReady }: { onReady: () => void }) {
  const { phase, checks, dismiss } = useBootCheck();
  const logoRef = useRef<HTMLDivElement>(null);
  const [logoPlayed, setLogoPlayed] = useState(false);
  const [exiting, setExiting] = useState(false);
  const [hintIdx, setHintIdx] = useState(0);

  /* ── GSAP logo entrance ──────────────────────────── */
  useEffect(() => {
    const el = logoRef.current;
    if (!el) return;
    const img = el.querySelector('.hatad-logo-img');
    const tl = gsap.timeline({ onComplete: () => setLogoPlayed(true) });
    if (img) {
      tl.fromTo(img,
        { opacity: 0, y: 10 },
        { opacity: 1, y: 0, duration: 0.6, ease: 'power2.out' },
      );
    } else {
      tl.to(el, { opacity: 1, duration: 0.4 });
    }
    return () => { tl.kill(); };
  }, []);

  /* ── Cycle hint text while loading ───────────────── */
  useEffect(() => {
    if (phase === 'ready' || phase === 'degraded' || phase === 'failed') return;
    const t = setInterval(() => setHintIdx(i => (i + 1) % HINTS.length), 2400);
    return () => clearInterval(t);
  }, [phase]);

  /* ── Auto-proceed (with welcome pause) ────────────── */
  useEffect(() => {
    if (!logoPlayed) return;
    if (phase === 'ready') {
      const t = setTimeout(() => setExiting(true), 3000);
      return () => clearTimeout(t);
    }
    if (phase === 'degraded') {
      const t = setTimeout(() => setExiting(true), 3000);
      return () => clearTimeout(t);
    }
  }, [phase, logoPlayed]);

  /* ── Exit → mount main app ───────────────────────── */
  useEffect(() => {
    if (!exiting) return;
    const t = setTimeout(onReady, 420);
    return () => clearTimeout(t);
  }, [exiting, onReady]);

  /* ── Failed checks for error state ───────────────── */
  const failedChecks = DISPLAY_ORDER
    .filter(k => checks[k]?.status === 'fail')
    .map(k => FRIENDLY[k] ?? checks[k].label);

  const isLoading = phase === 'connecting' || phase === 'checking';
  const isReady = phase === 'ready' || phase === 'degraded';
  const isFailed = phase === 'failed';

  return (
    <AnimatePresence>
      {!exiting && (
        <motion.div
          className="bootloader"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35, ease: 'easeInOut' }}
        >
          <div className="bootloader__content">
            {/* Logo */}
            <div className="bootloader__logo" ref={logoRef}>
              <HatadLogo className="bootloader__logo-inner" />
            </div>

            {/* Tagline */}
            <motion.p
              className="bootloader__tagline"
              initial={{ opacity: 0 }}
              animate={{ opacity: logoPlayed ? 1 : 0 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              Land Intelligence Platform
            </motion.p>

            {/* Loading state — progress bar + hint */}
            {logoPlayed && isLoading && (
              <motion.div
                className="bootloader__loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="bootloader__bar">
                  <div className="bootloader__bar-fill" />
                </div>
                <AnimatePresence mode="wait">
                  <motion.span
                    className="bootloader__hint"
                    key={hintIdx}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.25 }}
                  >
                    {HINTS[hintIdx]}
                  </motion.span>
                </AnimatePresence>
              </motion.div>
            )}

            {/* Welcome — all systems go */}
            {logoPlayed && isReady && (
              <motion.div
                className="bootloader__welcome"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: 'easeOut' }}
              >
                <svg className="bootloader__check-icon" viewBox="0 0 24 24" fill="none">
                  <motion.circle
                    cx="12" cy="12" r="10"
                    stroke="currentColor" strokeWidth="1.5"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                  />
                  <motion.path
                    d="M8 12.5l2.5 2.5 5.5-5.5"
                    stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.35, delay: 0.45, ease: 'easeOut' }}
                  />
                </svg>
                <span>All systems ready</span>
              </motion.div>
            )}

            {/* Error state — only on critical failure */}
            {logoPlayed && isFailed && (
              <motion.div
                className="bootloader__error"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <p className="bootloader__error-msg">
                  {failedChecks.length > 0
                    ? `Unable to reach ${failedChecks.join(', ')}`
                    : 'Unable to connect to the server'}
                </p>
                <p className="bootloader__error-sub">
                  This page will retry automatically.
                </p>
                <button
                  className="bootloader__skip-btn"
                  onClick={() => { dismiss(); setExiting(true); }}
                >
                  Continue anyway
                </button>
              </motion.div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
