import type { CheckResult } from '../../types';
import { StatusBadge, SeverityBadge } from '../common/StatusBadge';
import './ChecksTable.css';

interface Props {
  checks: CheckResult[];
}

export function ChecksTable({ checks }: Props) {
  const withEvidence = checks.filter(c => c.evidence && c.evidence.length >= 10 && !c.unverified);
  const unverified = checks.filter(c => c.unverified);
  const unreliable = checks.filter(c => c.unreliable);
  const withWarnings = checks.filter(c => c.guardrail_warnings && c.guardrail_warnings.length > 0);
  const gtVerified = checks.filter(c => c.ground_truth?.verified);
  const gtMismatches = checks.filter(c => c.ground_truth?.mismatches && c.ground_truth?.mismatches?.length > 0);

  return (
    <div className="checks-view">
      {/* Evidence strip */}
      {checks.length > 0 && (
        <div className="checks-strip">
          <span className="checks-strip__item checks-strip__item--green">Evidenced: {withEvidence.length}/{checks.length}</span>
          {unverified.length > 0 && (
            <span className="checks-strip__item checks-strip__item--amber">Unverified: {unverified.length}</span>
          )}
          {unreliable.length > 0 && (
            <span className="checks-strip__item checks-strip__item--amber">Unreliable: {unreliable.length}</span>
          )}
          {withWarnings.length > 0 && (
            <span className="checks-strip__item checks-strip__item--red">Guardrail Warnings: {withWarnings.length}</span>
          )}
          <span className="checks-strip__item checks-strip__item--blue">Ground-truth checked: {gtVerified.length}</span>
          {gtMismatches.length > 0 && (
            <span className="checks-strip__item checks-strip__item--red">Mismatches: {gtMismatches.length}</span>
          )}
        </div>
      )}

      <div className="checks-table-wrap">
        <table className="checks-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Severity</th>
              <th>Check</th>
              <th>Details & Evidence</th>
            </tr>
          </thead>
          <tbody>
            {checks.map((check, i) => (
              <tr key={i} className={`${check.unverified ? 'checks-row--unverified' : ''} ${check.unreliable ? 'checks-row--unreliable' : ''}`}>
                <td>
                  <StatusBadge status={check.status} size="sm" />
                  {check.unverified && (
                    <div className="checks-unverified-label">UNVERIFIED</div>
                  )}
                  {check.unreliable && (
                    <div className="checks-unreliable-label">UNRELIABLE</div>
                  )}
                </td>
                <td><SeverityBadge severity={check.severity} size="sm" /></td>
                <td className="checks-name-cell">
                  {check.rule_name}
                  {check.ground_truth?.verified && (
                    <div className="checks-gt">
                      {check.ground_truth.mismatches.length === 0 ? (
                        <span className="checks-gt--match">GT MATCH</span>
                      ) : (
                        <span className="checks-gt--mismatch">GT MISMATCH</span>
                      )}
                    </div>
                  )}
                </td>
                <td className="checks-detail-cell">
                  <div className="checks-explanation">{check.explanation}</div>
                  {check.status !== 'PASS' && check.recommendation && (
                    <div className="checks-rec">{check.recommendation}</div>
                  )}
                  {check.evidence && check.evidence.length >= 10 && (
                    <div className="checks-evidence">{check.evidence}</div>
                  )}
                  {check.ground_truth?.verified && (check.ground_truth.matches.length > 0 || check.ground_truth.mismatches.length > 0) && (
                    <div className="checks-gt-details">
                      {check.ground_truth.matches.map((m, j) => (
                        <div key={`m${j}`} className="checks-gt-match">{m}</div>
                      ))}
                      {check.ground_truth.mismatches.map((m, j) => (
                        <div key={`mm${j}`} className="checks-gt-mismatch">{m}</div>
                      ))}
                    </div>
                  )}
                  {check.guardrail_warnings && check.guardrail_warnings.length > 0 && (
                    <div className="checks-guardrail-warnings">
                      {check.guardrail_warnings.map((w, j) => (
                        <div key={`gw${j}`} className="checks-guardrail-warning">{w}</div>
                      ))}
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
