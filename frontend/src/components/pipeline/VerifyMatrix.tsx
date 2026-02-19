import { VERIFY_GROUPS, type PipelineState } from '../../types';
import './VerifyMatrix.css';

interface Props {
  pipeline: PipelineState;
}

/* User-friendly group names */
const GROUP_LABELS: Record<number, { name: string; desc: string; icon: string }> = {
  1: { name: 'Encumbrance',   desc: 'EC document checks',         icon: 'assignment' },
  2: { name: 'Sale Deed',     desc: 'Sale deed validation',       icon: 'gavel' },
  3: { name: 'Cross-Check',   desc: 'Cross-document consistency', icon: 'compare_arrows' },
  4: { name: 'Ownership',     desc: 'Chain of title analysis',    icon: 'link' },
  5: { name: 'Final Review',  desc: 'Overall assessment',         icon: 'checklist' },
};

export function VerifyMatrix({ pipeline }: Props) {
  return (
    <div className="viz-card">
      <div className="viz-card__header">
        <span className="viz-card__title">HATAD VERIFICATION</span>
        <span className="viz-card__sub">5-Pass Due Diligence</span>
      </div>
      <div className="verify-grid">
        {VERIFY_GROUPS.map(g => {
          const status = pipeline.verifyPasses[g.id] || 'pending';
          const result = pipeline.verifyPassResults[g.id];
          const label = GROUP_LABELS[g.id] || { name: g.name, desc: '', icon: 'check' };
          return (
            <div className={`verify-card verify-card--${status}`} key={g.id}>
              <div className="verify-card__top">
                <span className="material-icons" style={{ fontSize: 16, color: 'inherit' }}>
                  {label.icon}
                </span>
                <span className={`verify-card__dot verify-card__dot--${status}`} />
              </div>
              <div className="verify-card__name">{label.name}</div>
              <div className="verify-card__checks">{g.checks} checks</div>
              <div className="verify-card__needs">{label.desc}</div>
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
