import type { TabId } from '../../types';
import './TabBar.css';

interface Props {
  tabs: TabId[];
  activeTab: TabId;
  processing: boolean;
  onChange: (tab: TabId) => void;
}

const TAB_LABELS: Record<string, string> = {
  pipeline: 'Pipeline',
  log: 'Log',
  summary: 'Summary',
  checks: 'Checks',
  chain: 'Chain',
  transactions: 'Transactions',
  identity: 'Identity',
  knowledge: 'Knowledge',
  report: 'Report',
};

export function TabBar({ tabs, activeTab, processing, onChange }: Props) {
  return (
    <nav className="tabbar">
      {tabs.map(tab => (
        <button
          key={tab}
          className={`tabbar__tab ${activeTab === tab ? 'tabbar__tab--active' : ''} ${tab === 'pipeline' && processing ? 'tabbar__tab--live' : ''}`}
          onClick={() => onChange(tab)}
        >
          {tab === 'pipeline' && processing && <span className="tabbar__tab-pulse" />}
          <span>{TAB_LABELS[tab] || tab}</span>
        </button>
      ))}
      <div className="tabbar__spacer" />
    </nav>
  );
}
