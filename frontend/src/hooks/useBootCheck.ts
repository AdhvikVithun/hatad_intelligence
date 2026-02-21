/* ═══════════════════════════════════════════════════
   HATAD — useBootCheck hook
   Polls /api/analyze/health/boot until all critical
   systems are ready (or permanently failed).
   ═══════════════════════════════════════════════════ */

import { useState, useEffect, useCallback, useRef } from 'react';

export type CheckStatus = 'ok' | 'degraded' | 'fail';

export interface SystemCheck {
  status: CheckStatus;
  label: string;
  message?: string;
  model?: string;
  enabled?: boolean;
  free_gb?: number;
  models?: number;
}

export interface BootResult {
  overall: CheckStatus;
  checks: Record<string, SystemCheck>;
}

export type BootPhase =
  | 'connecting'   // haven't reached backend yet
  | 'checking'     // backend replied, running sub-checks
  | 'ready'        // all critical systems ok
  | 'degraded'     // non-critical issues (sarvam, embeddings)
  | 'failed';      // critical system down

export interface BootState {
  phase: BootPhase;
  checks: Record<string, SystemCheck>;
  overallStatus: CheckStatus | null;
  error: string | null;
  attempt: number;
  dismiss: () => void;
}

const POLL_INTERVAL = 3000;
const MAX_CONNECT_ATTEMPTS = 20;  // 60s to connect to backend

// Which subsystems are CRITICAL (block boot)?
const CRITICAL_KEYS = ['backend', 'filesystem', 'ollama', 'reasoning_model'];

export function useBootCheck(): BootState {
  const [phase, setPhase] = useState<BootPhase>('connecting');
  const [checks, setChecks] = useState<Record<string, SystemCheck>>({});
  const [overallStatus, setOverallStatus] = useState<CheckStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const dismissed = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismiss = useCallback(() => {
    dismissed.current = true;
    setPhase('ready');
  }, []);

  const runCheck = useCallback(async () => {
    if (dismissed.current) return;

    setAttempt(prev => prev + 1);

    try {
      const res = await fetch('/api/analyze/health/boot', { signal: AbortSignal.timeout(8000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BootResult = await res.json();

      setChecks(data.checks);
      setOverallStatus(data.overall);
      setError(null);

      // Check critical subsystems
      const criticalFailed = CRITICAL_KEYS.some(
        k => data.checks[k]?.status === 'fail'
      );

      if (criticalFailed) {
        setPhase('failed');
        // Keep polling — user may fix the issue
        timerRef.current = setTimeout(runCheck, POLL_INTERVAL);
      } else if (data.overall === 'degraded') {
        setPhase('degraded');
        // Auto-proceed after a brief pause (user sees the status)
      } else {
        setPhase('ready');
      }
    } catch {
      setPhase('connecting');
      setError('Waiting for backend to come online…');
      setAttempt(prev => {
        if (prev >= MAX_CONNECT_ATTEMPTS) {
          setPhase('failed');
          setError('Cannot reach the HATAD backend. Is the server running?');
          setChecks({
            backend: { status: 'fail', label: 'Backend API', message: 'Connection refused — server may not be running' },
          });
          return prev;
        }
        // Retry
        timerRef.current = setTimeout(runCheck, POLL_INTERVAL);
        return prev;
      });
    }
  }, []);

  useEffect(() => {
    // Small initial delay so the logo animation plays first
    timerRef.current = setTimeout(runCheck, 800);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [runCheck]);

  return { phase, checks, overallStatus, error, attempt, dismiss };
}
