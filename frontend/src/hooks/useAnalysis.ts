import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  uploadDocuments, startAnalysis, streamAnalysis,
  getResults, getSessions, checkLLMHealth, getReportPdfUrl,
} from '../api';
import type {
  DocFile, ProgressEntry, SessionData, PipelineState, Toast,
  SessionHistoryItem, TabId,
} from '../types';

// ═══════════════════════════════════════════════════
// Pipeline state derivation
// ═══════════════════════════════════════════════════

const STAGE_ORDER = [
  'extraction', 'classification', 'data_extraction',
  'knowledge', 'summarization', 'verification', 'report', 'complete',
];

function derivePipelineState(progress: ProgressEntry[]): PipelineState {
  const state: PipelineState = {
    currentStage: '',
    completedStages: new Set(),
    stageTimings: {},
    verifyPasses: { 1: 'pending', 2: 'pending', 3: 'pending', 4: 'pending', 5: 'pending' },
    verifyPassResults: {},
    llmCalls: 0,
    totalTokensIn: 0,
    totalTokensOut: 0,
    totalLlmTime: 0,
    lastLlmTask: '',
    lastLlmSpeed: 0,
    currentDataSize: 0,
    schemaEnforcedCalls: 0,
    thinkingCalls: 0,
    totalThinkingChars: 0,
    toolCallCount: 0,
    toolCallDetails: [],
    ragChunksIndexed: 0,
    ragDocsIndexed: 0,
    ragSearchCount: 0,
    mbFactCount: 0,
    mbConflictCount: 0,
    kbQueryCount: 0,
  };

  let prevStage = '';

  for (const entry of progress) {
    const stage = entry.stage;
    const msg = entry.message;
    const detail = entry.detail;

    if (STAGE_ORDER.includes(stage) && stage !== prevStage) {
      if (prevStage && prevStage !== 'complete') {
        state.completedStages.add(prevStage);
        if (state.stageTimings[prevStage]) state.stageTimings[prevStage].end = Date.now();
      }
      state.currentStage = stage;
      if (!state.stageTimings[stage]) state.stageTimings[stage] = { start: Date.now() };
      prevStage = stage;
    }

    if (detail?.type === 'llm_start') {
      state.llmCalls++;
      if (detail.prompt_tokens_est) state.totalTokensIn += detail.prompt_tokens_est;
      state.lastLlmTask = detail.task || '';
      if (detail.schema_enforced) state.schemaEnforcedCalls++;
      if (detail.thinking_enabled) state.thinkingCalls++;
    }
    if (detail?.type === 'llm_response') {
      if (detail.response_tokens) state.totalTokensOut += detail.response_tokens;
      if (detail.elapsed_seconds) state.totalLlmTime += detail.elapsed_seconds;
      if (detail.tokens_per_sec) state.lastLlmSpeed = detail.tokens_per_sec;
      if (detail.prompt_tokens) {
        state.totalTokensIn = state.totalTokensIn - (detail.prompt_tokens_est || 0) + detail.prompt_tokens;
      }
    }
    if (detail?.type === 'llm_thinking' && detail.thinking_chars) {
      state.totalThinkingChars += detail.thinking_chars;
    }
    if (detail?.type === 'llm_tool_call') {
      state.toolCallCount++;
      state.toolCallDetails.push({
        name: detail.tool_name || 'unknown',
        round: detail.round || 1,
        task: detail.task || '',
      });
    }
    if (detail?.type === 'llm_info' && detail.data_size) {
      state.currentDataSize = detail.data_size;
    }
    if (detail?.type === 'rag_stats') {
      state.ragChunksIndexed = detail.total_chunks || 0;
      state.ragDocsIndexed = detail.documents_indexed || 0;
    }
    if (detail?.type === 'llm_tool_call' && detail.tool_name === 'search_documents') {
      state.ragSearchCount++;
    }
    if (detail?.type === 'llm_tool_call' && detail.tool_name === 'query_knowledge_base') {
      state.kbQueryCount++;
    }
    if (detail?.type === 'memory_bank_summary') {
      state.mbFactCount = detail.total_facts || 0;
      state.mbConflictCount = detail.conflict_count || 0;
    }

    if (stage === 'verification') {
      const passMatch = msg.match(/Pass (\d)\/5/);
      if (passMatch) {
        const passId = parseInt(passMatch[1]);
        if (detail?.type === 'verify_group_done' && !detail.error) {
          // Successful completion (from detail)
          state.verifyPasses[passId] = 'done';
          state.verifyPassResults[passId] = {
            passed: detail.passed ?? 0,
            failed: detail.failed ?? 0,
            warned: detail.warnings ?? 0,
            deduction: detail.deduction ?? 0,
            thinking: detail.thinking || '',
          };
        } else if (detail?.type === 'verify_group_done' && detail.error) {
          // Group failed even after retry
          state.verifyPasses[passId] = 'error';
        } else if (msg.includes('done:') || msg.includes('Done')) {
          state.verifyPasses[passId] = 'done';
          const resultsMatch = msg.match(/(\d+) pass, (\d+) fail, (\d+) warn.*deduction: -(\d+)/);
          if (resultsMatch) {
            state.verifyPassResults[passId] = {
              passed: parseInt(resultsMatch[1]),
              failed: parseInt(resultsMatch[2]),
              warned: parseInt(resultsMatch[3]),
              deduction: parseInt(resultsMatch[4]),
              thinking: detail?.thinking || '',
            };
          }
        } else if (msg.includes('Skipped')) {
          state.verifyPasses[passId] = 'skipped';
        } else if (msg.includes('Failed')) {
          state.verifyPasses[passId] = 'error';
        } else if (msg.includes('Retry')) {
          state.verifyPasses[passId] = 'running';
        } else {
          state.verifyPasses[passId] = 'running';
        }
      }
      if (msg.includes('Meta Assessment')) state.verifyPasses[5] = 'running';
      if (msg.includes('All verification complete')) state.verifyPasses[5] = 'done';
    }

    if (stage === 'complete') {
      state.completedStages.add('verification');
      state.completedStages.add('report');
      state.completedStages.add('complete');
    }
  }

  return state;
}

// ═══════════════════════════════════════════════════
// Toast ID counter
// ═══════════════════════════════════════════════════
let _toastId = 0;

// ═══════════════════════════════════════════════════
// Main Hook
// ═══════════════════════════════════════════════════

export function useAnalysis() {
  // Core state
  const [files, setFiles] = useState<DocFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState<ProgressEntry[]>([]);
  const [session, setSession] = useState<SessionData | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('pipeline');
  const [llmOnline, setLlmOnline] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [clock, setClock] = useState('');
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryItem[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const progressEndRef = useRef<HTMLDivElement>(null);
  const startTimeRef = useRef<number>(0);

  // Toast helper
  const showToast = useCallback((type: Toast['type'], message: string) => {
    const id = ++_toastId;
    setToasts(prev => [...prev, { id, type, message }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // Clock
  useEffect(() => {
    const tick = () => {
      setClock(new Date().toLocaleTimeString('en-US', { hour12: false }));
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  // Elapsed timer
  useEffect(() => {
    if (!processing) return;
    startTimeRef.current = Date.now();
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [processing]);

  // LLM health
  useEffect(() => {
    checkLLMHealth().then(r => setLlmOnline(r.status === 'online'));
    const interval = setInterval(() => {
      checkLLMHealth().then(r => setLlmOnline(r.status === 'online'));
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll
  useEffect(() => {
    progressEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [progress]);

  // Pipeline state
  const pipeline = useMemo(() => derivePipelineState(progress), [progress]);

  // Session history
  const loadSessionHistory = useCallback(async () => {
    try {
      const data = await getSessions();
      setSessionHistory(data.sessions || []);
    } catch { /* ignore */ }
  }, []);

  const loadPastSession = useCallback(async (sid: string) => {
    try {
      const fullResults = await getResults(sid);
      setSession(fullResults);
      setSessionId(sid);
      setFiles(fullResults.documents || []);
      setProgress(fullResults.progress || []);
      setActiveTab('summary');
      showToast('info', `Loaded session ${sid.slice(0, 8)}...`);
    } catch (err: any) {
      showToast('error', `Failed to load session: ${err.message || 'Unknown error'}`);
    }
  }, [showToast]);

  // Handlers
  const handleUpload = useCallback(async (fileList: FileList | File[]) => {
    const pdfFiles = Array.from(fileList).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) {
      showToast('error', 'Only PDF files are accepted');
      return;
    }
    setUploading(true);
    try {
      const result = await uploadDocuments(pdfFiles);
      setFiles(prev => [...prev, ...result.uploaded]);
      showToast('success', `Uploaded ${result.uploaded.length} document(s)`);
    } catch (err: any) {
      showToast('error', `Upload failed: ${err?.response?.data?.detail || err.message || 'Unknown error'}`);
    } finally {
      setUploading(false);
    }
  }, [showToast]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    handleUpload(e.dataTransfer.files);
  }, [handleUpload]);

  const handleStartAnalysis = useCallback(async () => {
    if (files.length === 0) return;
    setProcessing(true);
    setProgress([]);
    setSession(null);
    setActiveTab('pipeline');
    setElapsed(0);

    try {
      const filenames = files.map(f => f.filename);
      const result = await startAnalysis(filenames);
      const sid = result.session_id;
      setSessionId(sid);

      streamAnalysis(
        sid,
        (data) => {
          setProgress(prev => [...prev, {
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
            stage: data.stage || '',
            message: data.message || '',
            detail: data.detail || undefined,
          }]);
          if (data.documents) setFiles(data.documents);
        },
        async () => {
          try {
            const fullResults = await getResults(sid);
            setSession(fullResults);
            if (fullResults?.incomplete) {
              showToast('info', 'Analysis completed with errors — some results may be incomplete.');
            }
          } catch {
            showToast('error', 'Analysis failed to complete. Check the pipeline log for details.');
          }
          setProcessing(false);
        },
        (err) => {
          showToast('error', `Analysis stream error: ${err}`);
          setProcessing(false);
        },
      );
    } catch (err: any) {
      showToast('error', `Analysis failed to start: ${err.message || 'Unknown error'}`);
      setProcessing(false);
    }
  }, [files, showToast]);

  const handleClear = useCallback(() => {
    if (files.length > 0 || session) {
      if (!window.confirm('Clear all documents and results? This cannot be undone.')) return;
    }
    setFiles([]);
    setSession(null);
    setSessionId(null);
    setProgress([]);
    setProcessing(false);
    setActiveTab('pipeline');
    setElapsed(0);
    showToast('info', 'Workspace cleared');
  }, [files, session, showToast]);

  // Derived data
  const verification = session?.verification_result;
  const riskScore = verification?.risk_score ?? null;
  const riskBand = verification?.risk_band ?? null;
  const checks = verification?.checks ?? [];
  const chain = verification?.chain_of_title ?? [];
  const redFlags = verification?.red_flags ?? [];
  const recommendations = verification?.recommendations ?? [];
  const missingDocs = verification?.missing_documents ?? [];

  const passCount = checks.filter(c => c.status === 'PASS').length;
  const failCount = checks.filter(c => c.status === 'FAIL').length;
  const warnCount = checks.filter(c => c.status === 'WARNING').length;
  const criticalChecks = checks.filter(c => c.severity === 'CRITICAL');

  const riskColor = riskBand === 'LOW' ? 'var(--green)' :
    riskBand === 'MEDIUM' ? 'var(--amber)' :
    riskBand === 'HIGH' ? 'var(--orange)' :
    riskBand === 'CRITICAL' ? 'var(--red)' : 'var(--text-dim)';

  const ecData = session?.extracted_data
    ? Object.values(session.extracted_data).find((d: any) => d.document_type === 'EC')
    : null;
  const transactions = (ecData as any)?.data?.transactions ?? [];

  const identityClusters = session?.identity_clusters ?? [];
  const memoryBank = session?.memory_bank ?? null;
  const mbFacts = memoryBank?.facts ?? [];
  const mbConflicts = memoryBank?.conflicts ?? [];
  const mbCrossRefs = memoryBank?.cross_references ?? [];
  const mbCategories = [...new Set(mbFacts.map(f => f.category))].sort();

  const lastLlmEvent = [...progress].reverse().find(p => p.detail?.type?.startsWith('llm_'));
  const llmActive = processing && lastLlmEvent?.detail?.type &&
    !['llm_done', 'llm_failed'].includes(lastLlmEvent.detail.type);

  // Chain graph data
  const chainGraphData = useMemo(() => {
    if (!chain || chain.length === 0) return null;
    const nodeMap = new Map<string, { id: string; group: string; txnCount: number }>();
    const links: Array<{
      source: string; target: string; date: string;
      transaction_type: string; document_number: string;
      valid: boolean; notes?: string; sequence: number;
      docSource?: string;
    }> = [];

    chain.forEach((link) => {
      const fromId = (link.from ?? '').trim();
      const toId = (link.to ?? '').trim();
      if (!fromId || !toId) return;
      if (!nodeMap.has(fromId)) nodeMap.set(fromId, { id: fromId, group: 'intermediate', txnCount: 0 });
      if (!nodeMap.has(toId)) nodeMap.set(toId, { id: toId, group: 'intermediate', txnCount: 0 });
      nodeMap.get(fromId)!.txnCount++;
      nodeMap.get(toId)!.txnCount++;
      links.push({
        source: fromId, target: toId, date: link.date,
        transaction_type: link.transaction_type, document_number: link.document_number,
        valid: link.valid, notes: link.notes, sequence: link.sequence,
        docSource: link.source,
      });
    });

    const firstOwner = (chain[0]?.from ?? '').trim();
    const lastOwner = (chain[chain.length - 1]?.to ?? '').trim();
    if (firstOwner && nodeMap.has(firstOwner)) nodeMap.get(firstOwner)!.group = 'origin';
    if (lastOwner && nodeMap.has(lastOwner)) nodeMap.get(lastOwner)!.group = 'current';

    return { nodes: Array.from(nodeMap.values()), links };
  }, [chain]);

  const tabs: TabId[] = processing || progress.length > 0
    ? ['pipeline', 'log', 'summary', 'checks', 'chain', 'transactions', 'identity', 'knowledge', 'report']
    : ['summary', 'checks', 'chain', 'transactions', 'identity', 'knowledge', 'report'];

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return {
    // State
    files, uploading, sessionId, processing, progress, session,
    activeTab, llmOnline, toasts, clock, dragging, elapsed,
    sessionHistory, pipeline, tabs,
    // Derived
    verification, riskScore, riskBand, checks, chain, redFlags,
    recommendations, missingDocs, passCount, failCount, warnCount,
    criticalChecks, riskColor, transactions, identityClusters, memoryBank,
    mbFacts, mbConflicts, mbCrossRefs, mbCategories,
    lastLlmEvent, llmActive, chainGraphData,
    // Refs
    fileInputRef, progressEndRef,
    // Actions
    setActiveTab, setDragging, showToast, dismissToast,
    handleUpload, handleDrop, handleStartAnalysis, handleClear,
    loadSessionHistory, loadPastSession,
    // Utils
    formatDuration, getReportPdfUrl,
  };
}
