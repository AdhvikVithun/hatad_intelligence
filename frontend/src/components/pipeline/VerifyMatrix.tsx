import { VERIFY_GROUPS, type PipelineState } from '../../types';
import './VerifyMatrix.css';

interface Props {
  pipeline: PipelineState;
}

export function VerifyMatrix({ pipeline }: Props) {
  return (
    <div className="viz-card">
      <div className="viz-card__header">
        <span className="viz-card__title">VERIFICATION MATRIX</span>
        <span className="viz-card__sub">5-Pass Analysis</span>
      </div>
      <div className="verify-grid">
        {VERIFY_GROUPS.map(g => {
          const status = pipeline.verifyPasses[g.id] || 'pending';
          const result = pipeline.verifyPassResults[g.id];
          return (
            <div className={`verify-card verify-card--${status}`} key={g.id}>
              <div className="verify-card__top">
                <span className="verify-card__num">P{g.id}</span>
                <span className={`verify-card__dot verify-card__dot--${status}`} />
              </div>
              <div className="verify-card__name">{g.name}</div>
              <div className="verify-card__checks">{g.checks} checks</div>
              <div className="verify-card__needs">{g.needs}</div>
              {status === 'running' && (
                <div className="verify-card__bar">
                  <div className="verify-card__bar-fill" />
                </div>
              )}
              {result && (
                <div className="verify-card__result">
                  <span className="vr-pass">{result.passed}✓</span>
                  <span className="vr-fail">{result.failed}✗</span>
                  <span className="vr-warn">{result.warned}!</span>
                  <span className="vr-ded">-{result.deduction}</span>
                </div>
              )}
              {result?.thinking && (
                <div className="verify-card__thinking" title={result.thinking}>
                  <span className="thinking-label">CoT</span>
                  <span className="thinking-text">{result.thinking.substring(0, 80)}...</span>
                </div>
              )}
              {status === 'skipped' && (
                <div className="verify-card__skip">SKIPPED</div>
              )}
              {status === 'error' && (
                <div className="verify-card__error">ERROR</div>
              )}
              {status === 'done' && !result && (
                <div className="verify-card__done">
                  <span className="material-icons" style={{fontSize: 16, color: 'var(--green)'}}>check_circle</span>
                  DONE
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
