import type { PipelineState } from '../../types';
import './KnowledgeBase.css';

interface Props {
  processing: boolean;
  pipeline: PipelineState;
  memoryBank: any | null;
  mbFacts: any[];
  mbConflicts: any[];
  mbCrossRefs: any[];
  mbCategories: string[];
}

export function KnowledgeBase({
  processing, pipeline, memoryBank,
  mbFacts, mbConflicts, mbCrossRefs, mbCategories,
}: Props) {
  return (
    <div className="kb-view">
      {/* ── Stats ── */}
      <div className="kb-stats">
        <KBStat label="Documents" value={pipeline.ragDocsIndexed || (memoryBank?.ingested_files?.length ?? 0)} />
        <KBStat label="Memory Facts" value={pipeline.mbFactCount || mbFacts.length} />
        <KBStat label="Text Chunks" value={pipeline.ragChunksIndexed} />
        <KBStat label="LLM Queries" value={pipeline.ragSearchCount + pipeline.kbQueryCount} />
        <KBStat
          label="Conflicts"
          value={pipeline.mbConflictCount || mbConflicts.length}
          color={(pipeline.mbConflictCount || mbConflicts.length) > 0 ? 'var(--red)' : 'var(--green)'}
        />
      </div>

      {/* ── How it works ── */}
      <div className="kb-arch">
        <div className="kb-arch__title">HOW IT WORKS</div>
        <div className="kb-arch__flow">
          <div className="kb-arch__step">
            <div className="kb-arch__num">01</div>
            <div className="kb-arch__name">MEMORY</div>
            <div className="kb-arch__desc">Structured facts extracted from every document</div>
          </div>
          <div className="kb-arch__arrow">+</div>
          <div className="kb-arch__step">
            <div className="kb-arch__num">02</div>
            <div className="kb-arch__name">KNOWLEDGE</div>
            <div className="kb-arch__desc">Full document text indexed for semantic search</div>
          </div>
          <div className="kb-arch__arrow">→</div>
          <div className="kb-arch__step kb-arch__step--active">
            <div className="kb-arch__num">03</div>
            <div className="kb-arch__name">VERIFY</div>
            <div className="kb-arch__desc">LLM queries both sources to cross-verify documents</div>
          </div>
        </div>
      </div>

      {/* ── Conflicts ── */}
      {mbConflicts.length > 0 && (
        <div className="kb-section">
          <div className="kb-section__title kb-section__title--red">CONFLICTS DETECTED</div>
          {mbConflicts.map((c: any, i: number) => (
            <div key={i} className="kb-conflict">
              <div className={`kb-conflict__severity kb-conflict__severity--${c.severity?.toLowerCase()}`}>{c.severity}</div>
              <div className="kb-conflict__desc">{c.description}</div>
              <div className="kb-conflict__details">
                <span>{c.fact_a?.source}: {c.fact_a?.key} = {JSON.stringify(c.fact_a?.value)}</span>
                <span className="kb-conflict__vs">vs</span>
                <span>{c.fact_b?.source}: {c.fact_b?.key} = {JSON.stringify(c.fact_b?.value)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Cross-references ── */}
      {mbCrossRefs.length > 0 && (
        <div className="kb-section">
          <div className="kb-section__title">CROSS-DOCUMENT REFERENCES</div>
          <table className="kb-table">
            <thead>
              <tr><th>Field</th><th>Sources</th><th>Status</th><th>Values</th></tr>
            </thead>
            <tbody>
              {mbCrossRefs.map((cr: any, i: number) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>{cr.key}</td>
                  <td>{cr.sources?.join(', ')}</td>
                  <td>
                    <span className={`kb-status ${cr.consistent ? 'kb-status--pass' : 'kb-status--fail'}`}>
                      {cr.consistent ? 'CONSISTENT' : 'MISMATCH'}
                    </span>
                  </td>
                  <td style={{ fontSize: 11 }}>{cr.values?.map((v: any) => JSON.stringify(v)).join(' | ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Facts ── */}
      {mbFacts.length > 0 && (
        <div className="kb-section">
          <div className="kb-section__title">MEMORY — STORED FACTS BY CATEGORY</div>
          {mbCategories.map(cat => {
            const catFacts = mbFacts.filter(f => f.category === cat);
            return (
              <div key={cat} className="kb-category">
                <div className="kb-category__header">
                  {cat.toUpperCase()} <span className="kb-category__count">{catFacts.length}</span>
                </div>
                <table className="kb-table kb-table--compact">
                  <thead><tr><th>Key</th><th>Value</th><th>Source</th><th>Conf</th></tr></thead>
                  <tbody>
                    {catFacts.map((f, i) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 500 }}>{f.key}</td>
                        <td className="kb-fact-value">
                          {typeof f.value === 'object' ? JSON.stringify(f.value) : String(f.value)}
                        </td>
                        <td style={{ fontSize: 10, color: 'var(--text-dim)' }}>{f.source}</td>
                        <td style={{ textAlign: 'center' }}>{(f.confidence * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      )}

      {/* ── LLM Query History ── */}
      {pipeline.toolCallDetails.filter(t => t.name === 'search_documents' || t.name === 'query_knowledge_base').length > 0 && (
        <div className="kb-section">
          <div className="kb-section__title">LLM QUERY HISTORY</div>
          <table className="kb-table kb-table--compact">
            <thead>
              <tr><th>#</th><th>Tool</th><th>Query Context</th><th>Round</th></tr>
            </thead>
            <tbody>
              {pipeline.toolCallDetails
                .filter(t => t.name === 'search_documents' || t.name === 'query_knowledge_base')
                .map((t, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td style={{ fontSize: 10, color: t.name === 'query_knowledge_base' ? 'var(--primary)' : 'var(--green)' }}>
                      {t.name === 'query_knowledge_base' ? 'MEMORY' : 'KNOWLEDGE'}
                    </td>
                    <td style={{ fontSize: 11 }}>{t.task}</td>
                    <td>{t.round}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Empty States ── */}
      {mbFacts.length === 0 && pipeline.ragChunksIndexed === 0 && !processing && (
        <div className="kb-empty">
          Knowledge base was not built for this session.<br />
          Upload documents and run analysis to populate memory and knowledge.
        </div>
      )}
      {mbFacts.length === 0 && pipeline.ragChunksIndexed === 0 && processing && pipeline.currentStage !== 'knowledge' && (
        <div className="kb-empty">
          Knowledge base will be built after document extraction.
        </div>
      )}
    </div>
  );
}

function KBStat({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="kb-stat-card">
      <div className="kb-stat-card__value" style={color ? { color } : undefined}>{value}</div>
      <div className="kb-stat-card__label">{label}</div>
    </div>
  );
}
