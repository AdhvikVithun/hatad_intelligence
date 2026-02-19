import type { PipelineState } from '../../types';
import './KnowledgeMonitor.css';

interface Props {
  pipeline: PipelineState;
  processing: boolean;
}

export function KnowledgeMonitor({ pipeline, processing }: Props) {
  if (pipeline.ragChunksIndexed === 0 && pipeline.mbFactCount === 0 && pipeline.currentStage !== 'knowledge') {
    return null;
  }

  return (
    <div className="viz-card">
      <div className="viz-card__header">
        <span className="viz-card__title">KNOWLEDGE & MEMORY</span>
        <span className="viz-card__sub">
          {pipeline.currentStage === 'knowledge' ? (
            <><span className="live-dot" /> BUILDING</>
          ) : (pipeline.ragChunksIndexed > 0 || pipeline.mbFactCount > 0) ? 'READY' : 'PENDING'}
        </span>
      </div>
      <div className="knowledge-grid">
        <div className="knowledge-stat">
          <div className="knowledge-stat__value">{pipeline.mbFactCount}</div>
          <div className="knowledge-stat__label">Memory Facts</div>
        </div>
        <div className="knowledge-stat">
          <div className="knowledge-stat__value">{pipeline.ragChunksIndexed}</div>
          <div className="knowledge-stat__label">Text Chunks</div>
        </div>
        <div className="knowledge-stat">
          <div className="knowledge-stat__value">{pipeline.ragSearchCount + pipeline.kbQueryCount}</div>
          <div className="knowledge-stat__label">Queries</div>
        </div>
        <div className="knowledge-stat">
          <div className="knowledge-stat__value" style={{ color: pipeline.mbConflictCount > 0 ? 'var(--red)' : 'var(--green)' }}>
            {pipeline.mbConflictCount}
          </div>
          <div className="knowledge-stat__label">Conflicts</div>
        </div>
      </div>
      {pipeline.mbConflictCount > 0 && (
        <div className="knowledge-alert">
          {pipeline.mbConflictCount} cross-document conflict{pipeline.mbConflictCount > 1 ? 's' : ''} detected
        </div>
      )}
    </div>
  );
}
