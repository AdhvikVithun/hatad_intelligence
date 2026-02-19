import { useState } from 'react';
import type { IdentityCluster } from '../../types';
import './IdentityPanel.css';

interface Props {
  clusters: IdentityCluster[];
}

const BAND_CLASS: Record<string, string> = {
  HIGH: 'identity-badge--high',
  MODERATE: 'identity-badge--moderate',
  LOW: 'identity-badge--low',
  'VERY LOW': 'identity-badge--very-low',
};

const BAND_ICON: Record<string, string> = {
  HIGH: 'verified',
  MODERATE: 'check_circle',
  LOW: 'warning',
  'VERY LOW': 'error',
};

function qualityClass(q: number) {
  if (q >= 0.7) return 'identity-mention-quality--good';
  if (q >= 0.4) return 'identity-mention-quality--fair';
  return 'identity-mention-quality--poor';
}

function qualityLabel(q: number) {
  if (q >= 0.7) return 'Good';
  if (q >= 0.4) return 'Fair';
  return 'Poor';
}

export function IdentityPanel({ clusters }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (clusters.length === 0) {
    return (
      <div className="identity-panel">
        <div className="chain-view__empty">No identity data available</div>
      </div>
    );
  }

  const totalMentions = clusters.reduce((s, c) => s + c.mention_count, 0);
  const avgConfidence = clusters.reduce((s, c) => s + c.confidence, 0) / clusters.length;
  const uniqueSources = new Set(clusters.flatMap(c => c.source_files));
  const highConf = clusters.filter(c => c.confidence_band === 'HIGH').length;

  return (
    <div className="identity-panel">
      {/* Stats strip */}
      <div className="identity-stats">
        <div className="identity-stat">
          <div className="identity-stat__value">{clusters.length}</div>
          <div className="identity-stat__label">Identities</div>
        </div>
        <div className="identity-stat">
          <div className="identity-stat__value">{totalMentions}</div>
          <div className="identity-stat__label">Mentions</div>
        </div>
        <div className="identity-stat">
          <div className="identity-stat__value">{uniqueSources.size}</div>
          <div className="identity-stat__label">Sources</div>
        </div>
        <div className="identity-stat">
          <div className="identity-stat__value">{Math.round(avgConfidence * 100)}%</div>
          <div className="identity-stat__label">Avg Confidence</div>
        </div>
        <div className="identity-stat">
          <div className="identity-stat__value">{highConf}</div>
          <div className="identity-stat__label">High Confidence</div>
        </div>
      </div>

      {/* Cluster cards */}
      {clusters.map(cluster => {
        const isOpen = expanded.has(cluster.cluster_id);
        const bandClass = BAND_CLASS[cluster.confidence_band] ?? 'identity-badge--low';
        const bandIcon = BAND_ICON[cluster.confidence_band] ?? 'help';

        return (
          <div className="identity-cluster" key={cluster.cluster_id}>
            {/* Header */}
            <div className="identity-cluster__header" onClick={() => toggle(cluster.cluster_id)}>
              <span className="material-icons identity-cluster__icon">person</span>
              <div className="identity-cluster__info">
                <div className="identity-cluster__name">{cluster.consensus_name}</div>
                <div className="identity-cluster__meta">
                  <span className="identity-cluster__meta-item">
                    <span className="material-icons">badge</span>
                    {cluster.cluster_id}
                  </span>
                  <span className="identity-cluster__meta-item">
                    <span className="material-icons">visibility</span>
                    {cluster.mention_count} mention{cluster.mention_count !== 1 ? 's' : ''}
                  </span>
                  <span className="identity-cluster__meta-item">
                    <span className="material-icons">description</span>
                    {cluster.source_files.length} source{cluster.source_files.length !== 1 ? 's' : ''}
                  </span>
                </div>
              </div>
              <span className={`identity-badge ${bandClass}`}>
                <span className="material-icons" style={{ fontSize: 13 }}>{bandIcon}</span>
                {Math.round(cluster.confidence * 100)}% {cluster.confidence_band}
              </span>
              <span className={`material-icons identity-cluster__expand ${isOpen ? 'identity-cluster__expand--open' : ''}`}>
                expand_more
              </span>
            </div>

            {/* Roles */}
            {cluster.roles.length > 0 && (
              <div className="identity-roles">
                {cluster.roles.map(role => (
                  <span className="identity-role-pill" key={role}>
                    <span className="material-icons" style={{ fontSize: 10 }}>assignment_ind</span>
                    {role}
                  </span>
                ))}
              </div>
            )}

            {/* Expanded body */}
            {isOpen && (
              <div className="identity-cluster__body">
                {/* Evidence */}
                {cluster.evidence_lines.length > 0 && (
                  <div className="identity-section">
                    <div className="identity-section__title">Evidence</div>
                    <ul className="identity-evidence">
                      {cluster.evidence_lines.map((line, i) => (
                        <li className="identity-evidence__item" key={i}>
                          <span className="material-icons identity-evidence__icon">arrow_right</span>
                          <span>{line}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Mentions table */}
                <div className="identity-section">
                  <div className="identity-section__title">Mentions</div>
                  <table className="identity-mentions-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Role</th>
                        <th>Source</th>
                        <th>Type</th>
                        <th>OCR Quality</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cluster.mentions.map((m, i) => (
                        <tr key={i}>
                          <td title={m.name}>{m.name}</td>
                          <td>{m.role}</td>
                          <td title={m.source_file}>{m.source_file}</td>
                          <td>{m.source_type}</td>
                          <td>
                            <span className={`identity-mention-quality ${qualityClass(m.ocr_quality)}`}>
                              {qualityLabel(m.ocr_quality)}
                              <span style={{ opacity: 0.6 }}>{Math.round(m.ocr_quality * 100)}%</span>
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Source files */}
                {cluster.source_files.length > 0 && (
                  <div className="identity-section">
                    <div className="identity-section__title">Source Documents</div>
                    <div className="identity-sources">
                      {cluster.source_files.map(f => (
                        <span className="identity-source-chip" key={f}>
                          <span className="material-icons">description</span>
                          {f}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
