import { PIPELINE_STAGES, type PipelineState } from '../../types';
import './StageTracker.css';

interface Props {
  pipeline: PipelineState;
}

const STAGE_ICONS: Record<string, string> = {
  extraction: 'document_scanner',
  classification: 'category',
  data_extraction: 'schema',
  knowledge: 'hub',
  summarization: 'compress',
  verification: 'verified_user',
  report: 'description',
  complete: 'check_circle',
};

export function StageTracker({ pipeline }: Props) {
  const currentStage = PIPELINE_STAGES.find(s => s.id === pipeline.currentStage);
  const total = PIPELINE_STAGES.length;
  const done = pipeline.completedStages.size;
  const allDone = done >= total;
  const pct = Math.min((done / (total - 1)) * 100, 100);

  return (
    <div className={`stage-tracker ${allDone ? 'stage-tracker--complete' : ''}`}>
      {/* header */}
      <div className="stage-tracker__header">
        <div className="stage-tracker__brand">
          <span className="material-icons stage-tracker__brand-icon">
            {allDone ? 'verified' : 'radar'}
          </span>
          <span className="stage-tracker__label">HATAD ANALYSIS PIPELINE</span>
        </div>
        <div className="stage-tracker__right">
          {allDone ? (
            <span className="stage-tracker__done-badge">
              <span className="material-icons" style={{ fontSize: 12 }}>check_circle</span>
              ALL STAGES COMPLETE
            </span>
          ) : (
            <span className="stage-tracker__progress">{done} of {total}</span>
          )}
        </div>
      </div>

      {/* rail */}
      <div className="stage-tracker__rail">
        <div className="stage-rail__line">
          <div className="stage-rail__fill" style={{ width: `${pct}%` }} />
        </div>

        {PIPELINE_STAGES.map((s, idx) => {
          const isComplete = pipeline.completedStages.has(s.id);
          const isCurrent = pipeline.currentStage === s.id && !isComplete;
          const status = isComplete ? 'done' : isCurrent ? 'active' : 'pending';

          return (
            <div className={`stage-node stage-node--${status}`} key={s.id}>
              {/* circle */}
              <div className="stage-node__circle">
                {isComplete ? (
                  <span className="material-icons stage-node__check">check</span>
                ) : (
                  <span className="material-icons stage-node__icon">
                    {STAGE_ICONS[s.id] || s.short}
                  </span>
                )}
              </div>

              {/* text */}
              <span className="stage-node__label">{s.label}</span>
              <span className="stage-node__desc">{s.desc}</span>
            </div>
          );
        })}
      </div>

      {/* active‑stage detail banner */}
      {currentStage && currentStage.id !== 'complete' && (
        <div className="stage-detail">
          <div className="stage-detail__icon-wrap">
            <span className="material-icons stage-detail__icon">
              {STAGE_ICONS[currentStage.id]}
            </span>
          </div>
          <div className="stage-detail__text">
            <span className="stage-detail__name">{currentStage.desc}</span>
            <span className="stage-detail__desc">{currentStage.detail}</span>
          </div>
          <span className="stage-detail__pulse" />
        </div>
      )}
    </div>
  );
}
