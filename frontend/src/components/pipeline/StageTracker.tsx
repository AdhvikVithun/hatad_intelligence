import { PIPELINE_STAGES, type PipelineState } from '../../types';
import './StageTracker.css';

interface Props {
  pipeline: PipelineState;
}

export function StageTracker({ pipeline }: Props) {
  return (
    <div className="stage-tracker">
      <div className="stage-tracker__header">
        <span className="stage-tracker__label">PIPELINE</span>
        <span className="stage-tracker__progress">
          {pipeline.completedStages.size} / {PIPELINE_STAGES.length} stages
        </span>
      </div>
      <div className="stage-tracker__rail">
        {/* Connector line behind all nodes */}
        <div className="stage-rail__line">
          <div
            className="stage-rail__fill"
            style={{
              width: `${(pipeline.completedStages.size / (PIPELINE_STAGES.length - 1)) * 100}%`,
            }}
          />
        </div>
        {/* Nodes */}
        {PIPELINE_STAGES.map((s) => {
          const isComplete = pipeline.completedStages.has(s.id);
          const isCurrent = pipeline.currentStage === s.id && !isComplete;
          const status = isComplete ? 'done' : isCurrent ? 'active' : 'pending';
          return (
            <div className={`stage-node stage-node--${status}`} key={s.id} title={s.desc}>
              <div className="stage-node__circle">
                {isComplete ? (
                  <span className="material-icons" style={{fontSize: 16}}>check</span>
                ) : (
                  <span>{s.short}</span>
                )}
              </div>
              <div className="stage-node__label">{s.label}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
