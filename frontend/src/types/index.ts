// ═══════════════════════════════════════════════════
// HATAD — Type Definitions
// ═══════════════════════════════════════════════════

export interface DocFile {
  filename: string;
  original_name: string;
  size: number;
  document_type?: string;
  classification?: any;
  pages?: number;
}

export interface LLMDetail {
  type: string;
  task?: string;
  model?: string;
  prompt_chars?: number;
  prompt_tokens_est?: number;
  prompt_tokens?: number;
  response_tokens?: number;
  tokens_per_sec?: number;
  elapsed_seconds?: number;
  temperature?: number;
  attempt?: number;
  error?: string;
  response_chars?: number;
  total_seconds?: number;
  data_size?: number;
  group?: number;
  image_count?: number;
  schema_enforced?: boolean;
  thinking_enabled?: boolean;
  thinking_chars?: number;
  thinking_preview?: string;
  tools_enabled?: boolean;
  tool_name?: string;
  tool_args?: Record<string, any>;
  result_preview?: string;
  round?: number;
  group_id?: number;
  group_name?: string;
  passed?: number;
  failed?: number;
  warnings?: number;
  deduction?: number;
  thinking?: string;
  total_chunks?: number;
  documents_indexed?: number;
  total_facts?: number;
  conflict_count?: number;
  cross_ref_count?: number;
}

export interface ProgressEntry {
  timestamp: string;
  stage: string;
  message: string;
  detail?: LLMDetail;
}

export interface CheckResult {
  rule_code: string;
  rule_name: string;
  severity: string;
  status: string;
  explanation: string;
  recommendation: string;
  evidence?: string;
  unverified?: boolean;
  unreliable?: boolean;
  data_confidence?: string;
  data_confidence_score?: number;
  guardrail_warnings?: string[];
  ground_truth?: {
    verified: boolean;
    matches: string[];
    mismatches: string[];
    skipped?: string[];
  };
}

export interface ChainLink {
  sequence: number;
  date: string;
  from: string;
  to: string;
  transaction_type: string;
  document_number: string;
  valid: boolean;
  notes?: string;
  transaction_id?: string;
}

// ── API response types ──────────────────────────────

export interface UploadResponse {
  uploaded: DocFile[];
  count: number;
  message: string;
}

export interface StartAnalysisResponse {
  session_id: string;
  status: string;
  message: string;
}

export interface SessionsResponse {
  sessions: SessionHistoryItem[];
}

export interface LLMHealthResponse {
  status: string;
  model?: string;
  error?: string;
}

export interface SessionData {
  session_id: string;
  status: string;
  incomplete?: boolean;
  risk_score: number | null;
  risk_band: string | null;
  documents: DocFile[];
  extracted_data: Record<string, any>;
  memory_bank: {
    facts: { category: string; key: string; value: any; source: string; confidence: number }[];
    conflicts: { fact_a: any; fact_b: any; description: string; severity: string }[];
    cross_references: { key: string; sources: string[]; consistent: boolean; values: any[] }[];
    ingested_files: string[];
    summary: Record<string, number>;
  } | null;
  verification_result: {
    risk_score: number;
    risk_band: string;
    executive_summary: string;
    checks: CheckResult[];
    chain_of_title: ChainLink[];
    red_flags: string[];
    recommendations: string[];
    missing_documents: string[];
    group_results_summary?: Record<string, { name: string; check_count: number; deduction: number }>;
  } | null;
  identity_clusters: IdentityCluster[] | null;
  narrative_report: string | null;
  progress: ProgressEntry[];
}

export interface PipelineState {
  currentStage: string;
  completedStages: Set<string>;
  stageTimings: Record<string, { start: number; end?: number }>;
  verifyPasses: Record<number, 'pending' | 'running' | 'done' | 'error' | 'skipped'>;
  verifyPassResults: Record<number, { passed: number; failed: number; warned: number; deduction: number; thinking: string }>;
  llmCalls: number;
  totalTokensIn: number;
  totalTokensOut: number;
  totalLlmTime: number;
  lastLlmTask: string;
  lastLlmSpeed: number;
  currentDataSize: number;
  schemaEnforcedCalls: number;
  thinkingCalls: number;
  totalThinkingChars: number;
  toolCallCount: number;
  toolCallDetails: { name: string; round: number; task: string }[];
  ragChunksIndexed: number;
  ragDocsIndexed: number;
  ragSearchCount: number;
  mbFactCount: number;
  mbConflictCount: number;
  kbQueryCount: number;
}

export interface Toast {
  id: number;
  type: 'error' | 'success' | 'info';
  message: string;
}

export interface SessionHistoryItem {
  session_id: string;
  status: string;
  risk_score: number | null;
  risk_band: string | null;
  created_at: string;
  documents: string[];
}

// Constants
export const PIPELINE_STAGES = [
  { id: 'extraction',      label: 'SCAN',      short: '1', desc: 'Document Scanning',         detail: 'HATAD OCR is reading and digitising your documents page by page.' },
  { id: 'classification',  label: 'IDENTIFY',  short: '2', desc: 'Document Identification',   detail: 'HATAD Intelligence is identifying each document type — Sale Deed, EC, Patta, and more.' },
  { id: 'data_extraction', label: 'ANALYSE',   short: '3', desc: 'Data Extraction',           detail: 'HATAD Intelligence is extracting key fields — names, survey numbers, dates, and amounts.' },
  { id: 'knowledge',       label: 'CONNECT',   short: '4', desc: 'Cross-Reference',           detail: 'Building a knowledge graph to cross-reference facts across all your documents.' },
  { id: 'summarization',   label: 'SUMMARISE', short: '5', desc: 'Document Summarisation',    detail: 'Compressing document data into concise summaries for verification.' },
  { id: 'verification',    label: 'VERIFY',    short: '6', desc: 'Due Diligence Checks',      detail: 'Running multi-pass verification — ownership chains, encumbrances, and cross-document consistency.' },
  { id: 'report',          label: 'REPORT',    short: '7', desc: 'Report Generation',         detail: 'Generating your comprehensive due diligence narrative report.' },
  { id: 'complete',        label: 'DONE',      short: '✓', desc: 'Analysis Complete',         detail: 'All checks are complete. Review your results below.' },
] as const;

export const VERIFY_GROUPS = [
  { id: 1, name: 'EC-Only', checks: 5, icon: '[1]', needs: 'EC' },
  { id: 2, name: 'Sale Deed', checks: 4, icon: '[2]', needs: 'SALE_DEED' },
  { id: 3, name: 'Cross-Doc Property', checks: 6, icon: '[3]', needs: 'EC+PATTA+SD' },
  { id: 4, name: 'Compliance', checks: 6, icon: '[4]', needs: 'EC+PATTA+SD' },
  { id: 5, name: 'Chain & Pattern', checks: 10, icon: '[5]', needs: 'EC+SD' },
  { id: 6, name: 'Meta', checks: 2, icon: '[M]', needs: 'Results 1-5' },
] as const;

export type TabId = 'pipeline' | 'log' | 'summary' | 'checks' | 'chain' | 'transactions' | 'identity' | 'knowledge' | 'report';

export interface IdentityMention {
  name: string;
  role: string;
  source_file: string;
  source_type: string;
  date: string | null;
  ocr_quality: number;
  given: string;
  patronymic: string;
}

export interface IdentityCluster {
  cluster_id: string;
  consensus_name: string;
  confidence: number;
  confidence_band: string;
  evidence_lines: string[];
  roles: string[];
  source_files: string[];
  source_types: string[];
  mention_count: number;
  mentions: IdentityMention[];
}
