import type { ChainLink } from '../../types';
import './ChainView.css';

interface Props {
  chain: ChainLink[];
  chainGraphData: any;
}

export function ChainView({ chain, chainGraphData }: Props) {
  if (chain.length === 0) {
    return <div className="chain-view__empty">No chain of title data available</div>;
  }

  return (
    <div className="chain-view">
      {chainGraphData && (
        <>
          {/* Legend */}
          <div className="chain-legend">
            <span><span className="chain-legend__dot" style={{ background: '#10b981' }} /> Origin Owner</span>
            <span><span className="chain-legend__dot" style={{ background: 'var(--primary)' }} /> Intermediate</span>
            <span><span className="chain-legend__dot" style={{ background: '#f59e0b' }} /> Current Owner</span>
            <span className="chain-legend__sep" />
            <span className="chain-legend__count">
              {chainGraphData.nodes.length} parties · {chainGraphData.links.length} transfers
            </span>
          </div>

          {/* Horizontal Node Diagram */}
          <div className="chain-nodes">
            {chainGraphData.nodes.map((node: any, i: number) => {
              const group = node.group as string;
              const isOrigin = group === 'origin';
              const isCurrent = group === 'current';
              const nodeClass = isOrigin ? 'chain-node--origin' : isCurrent ? 'chain-node--current' : 'chain-node--intermediate';
              
              // Find link connecting this node to next
              const outLink = chainGraphData.links.find((l: any) => {
                const src = typeof l.source === 'object' ? l.source.id : l.source;
                return src === node.id;
              });

              return (
                <div className="chain-node-group" key={node.id}>
                  <div className={`chain-node ${nodeClass}`}>
                    <span className="material-icons chain-node__icon">
                      {isOrigin ? 'account_circle' : isCurrent ? 'person_pin' : 'person'}
                    </span>
                    <div className="chain-node__name">{node.id}</div>
                    <div className="chain-node__role">
                      {isOrigin ? 'Origin' : isCurrent ? 'Current' : `${node.txnCount} txn`}
                    </div>
                  </div>
                  {i < chainGraphData.nodes.length - 1 && outLink && (
                    <div className={`chain-edge ${outLink.valid ? 'chain-edge--valid' : 'chain-edge--invalid'}`}>
                      <div className="chain-edge__line" />
                      <span className="material-icons chain-edge__arrow">arrow_forward</span>
                      <div className="chain-edge__label">
                        {outLink.transaction_type}
                        <span className="chain-edge__date">{outLink.date}</span>
                      </div>
                    </div>
                  )}
                  {i < chainGraphData.nodes.length - 1 && !outLink && (
                    <div className="chain-edge chain-edge--valid">
                      <div className="chain-edge__line" />
                      <span className="material-icons chain-edge__arrow">arrow_forward</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Timeline Table */}
          <div className="chain-timeline">
            <div className="chain-timeline__title">Transfer Timeline</div>
            <table className="chain-table">
              <thead>
                <tr>
                  <th>#</th><th>Date</th><th>From</th><th>To</th>
                  <th>Type</th><th>Document</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {chain.map((link, i) => (
                  <tr key={i} className={!link.valid ? 'chain-row--invalid' : ''}>
                    <td>{link.sequence}</td>
                    <td className="chain-cell--date">{link.date}</td>
                    <td className="chain-cell--from">{link.from}</td>
                    <td className="chain-cell--to">{link.to}</td>
                    <td>{link.transaction_type}</td>
                    <td className="chain-cell--doc">{link.document_number}</td>
                    <td>
                      {link.valid
                        ? <span className="chain-status--valid">Valid</span>
                        : <span className="chain-status--invalid">Invalid {link.notes && `- ${link.notes}`}</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
