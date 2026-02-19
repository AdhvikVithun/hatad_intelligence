import type { PipelineState, ProgressEntry } from '../../types';
import './StageCards.css';

interface Props {
  pipeline: PipelineState;
  progress: ProgressEntry[];
  processing: boolean;
}

/* ── helpers ── */
function countByStage(progress: ProgressEntry[], stage: string): number {
  return progress.filter(p => p.stage === stage).length;
}

function lastMessage(progress: ProgressEntry[], stage: string): string {
  const msgs = progress.filter(p => p.stage === stage);
  if (msgs.length === 0) return '';
  return msgs[msgs.length - 1].message;
}

function pagesScanned(progress: ProgressEntry[]): number {
  return progress.filter(p =>
    p.stage === 'extraction' && /page/i.test(p.message),
  ).length || countByStage(progress, 'extraction');
}

function docsClassified(progress: ProgressEntry[]): number {
  return progress.filter(p =>
    p.stage === 'classification' && /classified|identified|type/i.test(p.message),
  ).length;
}

function fieldsExtracted(progress: ProgressEntry[]): number {
  return progress.filter(p =>
    p.stage === 'data_extraction' && (p.detail?.type === 'llm_response' || /extract/i.test(p.message)),
  ).length;
}

function checksCompleted(pipeline: PipelineState): { done: number; total: number } {
  let done = 0;
  let total = 0;
  for (let i = 1; i <= 5; i++) {
    const s = pipeline.verifyPasses[i];
    if (s === 'done' || s === 'skipped' || s === 'error') done++;
    total++;
  }
  return { done, total };
}

/* ── Card definitions ── */
interface CardDef {
  id: string;
  icon: string;
  title: string;
  engine: string;
  stages: string[];              // which pipeline stages this card covers
}

const CARDS: CardDef[] = [
  {
    id: 'ocr',
    icon: 'document_scanner',
    title: 'Document Scanner',
    engine: 'HATAD OCR',
    stages: ['extraction'],
  },
  {
    id: 'intelligence',
    icon: 'psychology',
    title: 'AI Analysis',
    engine: 'HATAD Intelligence',
    stages: ['classification', 'data_extraction', 'summarization'],
  },
  {
    id: 'knowledge',
    icon: 'hub',
    title: 'Knowledge Engine',
    engine: 'HATAD Knowledge',
    stages: ['knowledge'],
  },
  {
    id: 'verify',
    icon: 'verified_user',
    title: 'Verification',
    engine: 'HATAD Verify',
    stages: ['verification'],
  },
  {
    id: 'report',
    icon: 'description',
    title: 'Report Writer',
    engine: 'HATAD Reports',
    stages: ['report'],
  },
];

export function StageCards({ pipeline, progress, processing }: Props) {
  const verify = checksCompleted(pipeline);

  return (
    <div className="stage-cards">
      {CARDS.map(card => {
        const isActive = card.stages.includes(pipeline.currentStage);
        const isDone = card.stages.every(s => pipeline.completedStages.has(s));
        const hasStarted = isDone || isActive || card.stages.some(
          s => progress.some(p => p.stage === s),
        );
        const status = isDone ? 'done' : isActive ? 'active' : hasStarted ? 'waiting' : 'pending';

        /* derive stat for each card */
        let stat = '';
        let subtext = '';
        if (card.id === 'ocr') {
          const n = pagesScanned(progress);
          stat = `${n}`;
          subtext = n === 1 ? 'page scanned' : 'pages scanned';
        } else if (card.id === 'intelligence') {
          const cls = docsClassified(progress);
          const ext = fieldsExtracted(progress);
          if (ext > 0) { stat = `${ext}`; subtext = 'extraction tasks'; }
          else if (cls > 0) { stat = `${cls}`; subtext = 'documents identified'; }
          else { stat = `${pipeline.llmCalls}`; subtext = 'analysis tasks'; }
        } else if (card.id === 'knowledge') {
          stat = `${pipeline.mbFactCount}`;
          subtext = 'facts indexed';
        } else if (card.id === 'verify') {
          stat = `${verify.done}/${verify.total}`;
          subtext = 'verification passes';
        } else if (card.id === 'report') {
          stat = isDone ? '✓' : '—';
          subtext = isDone ? 'report ready' : 'pending';
        }

        return (
          <div className={`stage-card stage-card--${status}`} key={card.id}>
            <div className="stage-card__header">
              <span className={`material-icons stage-card__icon stage-card__icon--${status}`}>
                {isDone ? 'check_circle' : card.icon}
              </span>
              <span className="stage-card__engine">{card.engine}</span>
              {isActive && <span className="stage-card__live"><span className="live-dot" /> LIVE</span>}
              {isDone && <span className="stage-card__done-badge">DONE</span>}
            </div>
            <div className="stage-card__title">{card.title}</div>
            {hasStarted && (
              <div className="stage-card__stat">
                <span className="stage-card__stat-value">{stat}</span>
                <span className="stage-card__stat-label">{subtext}</span>
              </div>
            )}
            {isActive && (
              <div className="stage-card__bar">
                <div className="stage-card__bar-fill" />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
